import cv2
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image
import time
import threading
import queue
import ctypes
from concurrent.futures import ThreadPoolExecutor


# 添加PyOpenGL相关库导入
# 确保 PyOpenGL 配置为使用 EGL (通常在 ARM Linux 上是必要的)
# 可能需要在运行脚本前设置环境变量: export PYOPENGL_PLATFORM=egl
try:
    from OpenGL.GL import *
    from OpenGL.GLES2 import * # 明确使用 GLES2
    # 注意：PyOpenGL 可能不直接提供 GLES3 的完整绑定，GLES2 更安全
    from OpenGL.GL.shaders import compileProgram, compileShader
    # GLUT 通常不用于 EGL/离屏渲染，GLFW 是更好的选择
    # from OpenGL.GLUT import *
    # from OpenGL.GLU import *
    import glfw
except ImportError:
    print("错误：缺少必要的 OpenGL 或 GLFW 库。请安装 PyOpenGL PyOpenGL_accelerate glfw")
    exit()


def fast_check_exposure(image, under_threshold=15, over_threshold=180):
    """
    更快的曝光检测算法
    """
    # 降低采样率，使用更小的图像
    h, w = image.shape[:2]
    if h > 720 or w > 1280:  # 使用更小的尺寸
        scale_factor = min(1280 / w, 720 / h)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        # 使用INTER_NEAREST更快
        small_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        small_img = image

    # 直接转换灰度图
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)

    # 使用numpy掩码操作替代直方图计算
    total_pixels = gray.size
    under_pixels = np.sum(gray < under_threshold)
    over_pixels = np.sum(gray > over_threshold)

    under_percentage = (under_pixels / total_pixels) * 100
    over_percentage = (over_pixels / total_pixels) * 100

    return under_percentage, over_percentage


def check_exposure(input_image, under_threshold=15, over_threshold=180, print_info=False):
    """
    同时检测图像的欠曝光和过曝光区域，提高处理效率
    """
    # 判断输入是路径还是图像数组
    if isinstance(input_image, str):
        image = cv2.imread(input_image)
        if image is None:
            raise ValueError(f"无法读取图像: {input_image}")
    else:
        image = input_image

    # 使用快速曝光检测
    under_percentage, over_percentage = fast_check_exposure(image, under_threshold, over_threshold)

    # 创建掩码仅当需要时
    under_binary_mask = None
    over_binary_mask = None

    if under_percentage > 0.1 or over_percentage > 0.1:
        # 只在需要掩码时才计算掩码
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if under_percentage > 0.1:
            under_binary_mask = (gray < under_threshold).astype(np.uint8) * 255
        if over_percentage > 0.1:
            over_binary_mask = (gray > over_threshold).astype(np.uint8) * 255

    if print_info:
        print(f"欠曝光区域: {under_percentage:.2f}%, 过曝光区域: {over_percentage:.2f}%")

    return under_percentage, over_percentage, under_binary_mask, over_binary_mask, image


# OpenGL 上下文管理器 - 专为ARM Mali-G610 MP4 GPU优化
class GPUContext:
    """GPU上下文管理器，用于初始化OpenGL环境和管理着色器"""

    def __init__(self):
        self.initialized = False
        self.window = None
        self.shader_programs = {}
        self.textures = {}
        self.framebuffers = {}

        # 伽马校正着色器代码
        self.gamma_vertex_shader = """
        #version 300 es
        precision mediump float;
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 texCoord;
        out vec2 fragTexCoord;

        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            fragTexCoord = texCoord;
        }
        """

        self.gamma_fragment_shader = """
        #version 300 es
        precision mediump float;
        in vec2 fragTexCoord;
        uniform sampler2D inputTexture;
        uniform float gamma;
        out vec4 outColor;

        void main() {
            vec4 color = texture(inputTexture, fragTexCoord);
            outColor = vec4(pow(color.rgb, vec3(1.0/gamma)), color.a);
        }
        """

        # SSR着色器代码
        self.ssr_fragment_shader = """
        #version 300 es
        precision mediump float;
        in vec2 fragTexCoord;
        uniform sampler2D inputTexture;
        uniform float scale;
        uniform float k;
        uniform int kernelSize;
        out vec4 outColor;

        // 计算高斯权重
        float gaussian(float x, float sigma) {
            return exp(-(x*x)/(2.0*sigma*sigma)) / (sigma * sqrt(2.0 * 3.14159265359));
        }

        void main() {
            vec2 texSize = vec2(textureSize(inputTexture, 0));
            vec3 color = texture(inputTexture, fragTexCoord).rgb;

            // 对数变换
            vec3 logColor = log(color + vec3(1.0));

            // 高斯模糊实现
            float sigma = scale / 2.0;
            float G_ratio = (sigma*sigma) / (sigma*sigma + k);

            vec3 blurredLog = vec3(0.0);
            float totalWeight = 0.0;

            for (int y = -kernelSize/2; y <= kernelSize/2; y++) {
                for (int x = -kernelSize/2; x <= kernelSize/2; x++) {
                    vec2 offset = vec2(float(x), float(y)) / texSize;
                    float weight = gaussian(length(vec2(x, y)), sigma);
                    blurredLog += texture(inputTexture, fragTexCoord + offset).rgb * weight;
                    totalWeight += weight;
                }
            }

            // 归一化模糊结果
            if (totalWeight > 0.0) {
                blurredLog /= totalWeight;
            }

            // 视网膜算法公式
            vec3 result = exp(blurredLog + G_ratio * (logColor - blurredLog)) - vec3(1.0);

            // 输出结果
            outColor = vec4(clamp(result, 0.0, 1.0), 1.0);
        }
        """

    def initialize(self):
        """初始化OpenGL环境，创建离屏窗口"""
        if self.initialized:
            return True

        try:
            # 初始化GLFW
            if not glfw.init():
                print("无法初始化GLFW")
                return False

            # 配置GLFW - 针对ARM Mali GPU优化
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)  # 降级到OpenGL ES 2.0
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_ANY_PROFILE)

            # 创建无窗口上下文
            self.window = glfw.create_window(640, 480, "GPUContext", None, None)
            if not self.window:
                print("无法创建GLFW窗口")
                glfw.terminate()
                return False

            # 激活上下文
            glfw.make_context_current(self.window)
            
            # 检查OpenGL版本和支持
            gl_version = glGetString(GL_VERSION)
            gl_vendor = glGetString(GL_VENDOR)
            gl_renderer = glGetString(GL_RENDERER)
            
            print(f"OpenGL版本: {gl_version.decode() if gl_version else 'unknown'}")
            print(f"OpenGL厂商: {gl_vendor.decode() if gl_vendor else 'unknown'}")
            print(f"OpenGL渲染器: {gl_renderer.decode() if gl_renderer else 'unknown'}")
            
            # 如果检测到不支持的GPU或驱动问题，提前返回
            if gl_renderer and b"llvmpipe" in gl_renderer:
                print("检测到软件渲染，不使用GPU加速")
                return False

            # 编译着色器 - 简化为OpenGL ES 2.0兼容版本
            self.gamma_vertex_shader = """
            attribute vec2 position;
            attribute vec2 texCoord;
            varying vec2 fragTexCoord;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                fragTexCoord = texCoord;
            }
            """

            self.gamma_fragment_shader = """
            precision mediump float;
            varying vec2 fragTexCoord;
            uniform sampler2D inputTexture;
            uniform float gamma;

            void main() {
                vec4 color = texture2D(inputTexture, fragTexCoord);
                gl_FragColor = vec4(pow(color.rgb, vec3(1.0/gamma)), color.a);
            }
            """

            # 编译着色器
            try:
                self.shader_programs['gamma'] = compileProgram(
                    compileShader(self.gamma_vertex_shader, GL_VERTEX_SHADER),
                    compileShader(self.gamma_fragment_shader, GL_FRAGMENT_SHADER)
                )
                
                # 创建顶点数据
                self.vertices = np.array([
                    -1.0, -1.0, 0.0, 0.0,
                    1.0, -1.0, 1.0, 0.0,
                    -1.0, 1.0, 0.0, 1.0,
                    1.0, 1.0, 1.0, 1.0
                ], dtype=np.float32)

                # 创建顶点缓冲区
                self.vbo = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
                glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
                
                self.initialized = True
                return True
                
            except Exception as e:
                print(f"着色器编译失败: {e}")
                return False
                
        except Exception as e:
            print(f"OpenGL初始化失败: {e}")
            return False

    def cleanup(self):
        """清理OpenGL资源"""
        if self.initialized:
            # 删除所有纹理和帧缓冲
            for tex_id in self.textures.values():
                glDeleteTextures(1, [tex_id])

            for fb_id in self.framebuffers.values():
                glDeleteFramebuffers(1, [fb_id])

            # 删除所有着色器程序
            for program in self.shader_programs.values():
                glDeleteProgram(program)

            glfw.destroy_window(self.window)
            glfw.terminate()
            self.initialized = False

    def get_texture(self, img, texture_name=None):
        """将OpenCV图像转换为OpenGL纹理"""
        if not self.initialized:
            if not self.initialize():
                return None

        # 生成纹理名称
        if texture_name is None:
            texture_name = f"tex_{len(self.textures)}"

        # 转换图像格式，OpenCV是BGR，而OpenGL需要RGB
        if img.shape[2] == 3:  # BGR格式
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:  # BGRA格式
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

        # 创建OpenGL纹理
        if texture_name in self.textures:
            # 更新现有纹理
            tex_id = self.textures[texture_name]
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img_rgb)
        else:
            # 创建新纹理
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img_rgb)
            self.textures[texture_name] = tex_id

        return tex_id

    def get_framebuffer(self, width, height, fb_name=None):
        """创建或获取帧缓冲区"""
        if not self.initialized:
            if not self.initialize():
                return None, None

        # 生成帧缓冲名称
        if fb_name is None:
            fb_name = f"fb_{len(self.framebuffers)}"

        # 检查是否已存在匹配尺寸的帧缓冲
        if fb_name in self.framebuffers:
            # 获取现有帧缓冲
            fb_id = self.framebuffers[fb_name]
            tex_id = self.textures.get(f"{fb_name}_tex")

            # 检查纹理尺寸是否需要更新
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, width_ptr := np.array([0], dtype=np.int32))
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, height_ptr := np.array([0], dtype=np.int32))

            if width_ptr[0] != width or height_ptr[0] != height:
                # 更新纹理尺寸
                glBindTexture(GL_TEXTURE_2D, tex_id)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        else:
            # 创建新的帧缓冲和纹理
            fb_id = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, fb_id)

            # 创建纹理附件
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_id, 0)

            # 检查帧缓冲完整性
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                print("帧缓冲区不完整!")
                return None, None

            self.framebuffers[fb_name] = fb_id
            self.textures[f"{fb_name}_tex"] = tex_id

        return fb_id, tex_id

    def run_gamma_correction(self, img, gamma):
        """使用GPU执行伽马校正，增强错误处理"""
        if not self.initialized and not self.initialize():
            return adjust_gamma_cpu(img, gamma)

        try:
            h, w = img.shape[:2]

            # 创建输入纹理
            input_tex = self.get_texture(img, "gamma_input")
            if input_tex is None:
                return adjust_gst_gamma_cpu(img, gamma)

            # 设置帧缓冲
            fb_id, output_tex = self.get_framebuffer(w, h, "gamma_output")
            if fb_id is None or output_tex is None:
                return adjust_gamma_cpu(img, gamma)
                
            glBindFramebuffer(GL_FRAMEBUFFER, fb_id)
            
            # 检查帧缓冲状态
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                print("帧缓冲不完整，回退到CPU处理")
                return adjust_gamma_cpu(img, gamma)
                
            glViewport(0, 0, w, h)

            # 使用伽马校正着色器
            shader_program = self.shader_programs['gamma']
            glUseProgram(shader_program)

            # 设置着色器参数
            glUniform1i(glGetUniformLocation(shader_program, "inputTexture"), 0)
            glUniform1f(glGetUniformLocation(shader_program, "gamma"), gamma)

            # 绑定输入纹理
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, input_tex)

            # 设置顶点属性
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

            # 位置属性
            position_loc = glGetAttribLocation(shader_program, "position")
            glEnableVertexAttribArray(position_loc)
            glVertexAttribPointer(position_loc, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))

            # 纹理坐标属性
            texcoord_loc = glGetAttribLocation(shader_program, "texCoord")
            glEnableVertexAttribArray(texcoord_loc)
            glVertexAttribPointer(texcoord_loc, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))

            # 渲染四边形
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

            # 禁用顶点属性
            glDisableVertexAttribArray(position_loc)
            glDisableVertexAttribArray(texcoord_loc)

            # 安全读取结果
            try:
                result_buffer = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
                
                # 转换为Numpy数组
                result_array = np.frombuffer(result_buffer, dtype=np.uint8).reshape(h, w, 3)
                
                # OpenGL坐标系与图像坐标系不同, 需垂直翻转
                result_array = np.flipud(result_array)
                
                # 转换回BGR
                result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
                
                return result_bgr
                
            except Exception as e:
                print(f"GPU读取结果失败: {e}")
                return adjust_gamma_cpu(img, gamma)

        except Exception as e:
            print(f"GPU伽马校正出错: {e}")
            # 回退到CPU模式
            return adjust_gamma_cpu(img, gamma)

    def run_ssr(self, img, scale=15, k=0.02):
        """使用GPU执行SSR算法"""
        if not self.initialized and not self.initialize():
            print("无法使用GPU, 回退到CPU模式")
            return fast_SSR_cpu(img, scale, k)

        try:
            h, w = img.shape[:2]

            # 创建输入纹理
            input_tex = self.get_texture(img, "ssr_input")

            # 设置帧缓冲
            fb_id, output_tex = self.get_framebuffer(w, h, "ssr_output")
            glBindFramebuffer(GL_FRAMEBUFFER, fb_id)
            glViewport(0, 0, w, h)

            # 使用SSR着色器
            shader_program = self.shader_programs['ssr']
            glUseProgram(shader_program)

            # 设置着色器参数
            glUniform1i(glGetUniformLocation(shader_program, "inputTexture"), 0)
            glUniform1f(glGetUniformLocation(shader_program, "scale"), float(scale))
            glUniform1f(glGetUniformLocation(shader_program, "k"), float(k))
            kernel_size = scale * 2 + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            glUniform1i(glGetUniformLocation(shader_program, "kernelSize"), kernel_size)

            # 绑定输入纹理
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, input_tex)

            # 设置顶点属性
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

            # 位置属性
            position_loc = 0
            glEnableVertexAttribArray(position_loc)
            glVertexAttribPointer(position_loc, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))

            # 纹理坐标属性
            texcoord_loc = 1
            glEnableVertexAttribArray(texcoord_loc)
            glVertexAttribPointer(texcoord_loc, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))

            # 渲染四边形
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

            # 禁用顶点属性
            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(1)

            # 读取结果
            glBindFramebuffer(GL_FRAMEBUFFER, fb_id)
            result_buffer = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)

            # 转换为Numpy数组
            result_array = np.frombuffer(result_buffer, dtype=np.uint8).reshape(h, w, 3)

            # OpenGL坐标系与图像坐标系不同, 需垂直翻转
            result_array = np.flipud(result_array)

            # 转换回BGR
            result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

            return result_bgr

        except Exception as e:
            print(f"GPU SSR处理出错: {e}")
            # 回退到CPU模式
            return fast_SSR_cpu(img, scale, k)


# 创建全局GPU上下文
gpu_context = None


def get_gpu_context():
    """获取全局GPU上下文"""
    global gpu_context
    if gpu_context is None:
        gpu_context = GPUContext()
        if not gpu_context.initialize():
            print("警告: GPU初始化失败，将使用CPU进行处理")
    return gpu_context


# 保留原始的CPU版本函数作为备用
def create_gamma_table(gamma):
    """预计算并返回伽马校正查找表"""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return table


# 预计算伽马校正查找表
GAMMA_TABLES = {}
for gamma in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
    GAMMA_TABLES[gamma] = create_gamma_table(gamma)


def adjust_gamma_cpu(image, gamma=2.0):
    """使用CPU的伽马校正函数（作为备用）"""
    # 量化gamma值到0.1的精度以提高查找表命中率
    gamma = round(gamma * 10) / 10

    # 使用预计算表
    if gamma in GAMMA_TABLES:
        table = GAMMA_TABLES[gamma]
    else:
        # 仅在必要时创建新表
        table = create_gamma_table(gamma)
        # 缓存新值以加速后续处理
        if len(GAMMA_TABLES) < 50:  # 限制缓存大小
            GAMMA_TABLES[gamma] = table

    # 应用查找表
    return cv2.LUT(image, table)


# 新的GPU加速伽马校正函数
def adjust_gamma(image, gamma=2.0):
    """更强健的伽马校正函数，优先尝试CPU处理"""
    # 首先尝试使用CPU处理（性能更可靠）
    if gamma == 1.0:  # 伽马=1.0时不需要处理
        return image.copy()
    
    try:
        # 尝试使用CPU处理
        return adjust_gamma_cpu(image, gamma)
    except Exception as e:
        print(f"CPU伽马校正失败: {e}")
        
        # 如果CPU处理失败，尝试GPU（不太可能发生）
        try:
            gpu_ctx = get_gpu_context()
            if gpu_ctx and gpu_ctx.initialized:
                return gpu_ctx.run_gamma_correction(image, gamma)
        except Exception as e:
            print(f"GPU伽马校正失败: {e}")
            
            # 如果两种方法都失败，返回原图
            return image.copy()


# CPU版本的SSR算法（作为备用）
def fast_SSR_cpu(img, scale=15, k=0.02):
    """单尺度视网膜色彩恢复算法的CPU版本（备用）"""
    # 降低分辨率到更小尺寸，显著提高速度
    h, w = img.shape[:2]
    max_size = 300  # 降低到300像素，处理速度更快
    if max(h, w) > max_size:
        scale_factor = max_size / max(h, w)
        small_img = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)))
    else:
        small_img = img

    # 转换为浮点并归一化，一次性转换所有通道
    img_float = small_img.astype(np.float32) / 255.0

    # 对数转换一次性处理所有通道
    img_log = np.log1p(img_float)
    kernel_size = scale * 2 + 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    sigma = scale / 2
    G_ratio = sigma ** 2 / (sigma ** 2 + k)

    # 并行处理所有通道
    result = np.zeros_like(img_float)

    # 创建高斯核
    ksize = (kernel_size, kernel_size)

    # 一次性处理所有通道
    for z in range(3):
        img_smooth = cv2.GaussianBlur(img_log[:, :, z], ksize, sigma)
        result[:, :, z] = np.exp(img_smooth + G_ratio * (img_log[:, :, z] - img_smooth)) - 1

    # 使用更高效的numpy操作
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    # 如果之前缩小了图像，恢复原始尺寸
    if max(h, w) > max_size:
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

    return result


# GPU加速版本的SSR算法
def fast_SSR(img, scale=15, k=0.02):
    """GPU加速的单尺度视网膜色彩恢复算法"""
    try:
        # 尝试使用GPU处理
        gpu_ctx = get_gpu_context()
        if gpu_ctx and gpu_ctx.initialized:
            return gpu_ctx.run_ssr(img, scale, k)
    except Exception as e:
        print(f"GPU SSR处理失败: {e}")

    # 如果GPU处理失败，回退到CPU版本
    return fast_SSR_cpu(img, scale, k)


# 修改process_single_image函数以使用GPU加速
def process_single_image(image_path, over_threshold=220, under_threshold=15, output_dir='./outputs', gamma=1.8,
                         scales=[15], k=0.02, print_info=False):
    """使用GPU加速处理单个图像文件"""
    # 获取基本文件名
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_fixed{ext}")

    # 检测过曝和欠曝区域
    under_percentage, over_percentage, under_exposure_mask, over_exposure_mask, image = check_exposure(
        image_path, under_threshold, over_threshold, print_info=print_info)

    # 仅在需要时进行处理
    if under_percentage > 0.1 or over_percentage > 0.1:
        processed_image = image.copy()

        if under_percentage - over_percentage > 0.1:
            # 对欠曝区域应用低光照增强
            gamma_value = 1.1 + under_percentage / 100
            processed_image = adjust_gamma(processed_image, gamma=gamma_value)
        elif over_percentage - under_percentage > 0.1:
            # 对过曝区域处理
            gamma_value = max(0.5, 0.8 - over_percentage / 200)
            processed_image = adjust_gamma(processed_image, gamma=gamma_value)

            # 对严重过曝区域应用SSR算法
            if over_percentage > 20:
                processed_image = fast_SSR(processed_image, scale=scales[0], k=k)

        cv2.imwrite(output_path, processed_image)
        return output_path, over_percentage, under_percentage

    return None, over_percentage, under_percentage


def process_image(image_path='./inputs', over_threshold=220, under_threshold=15, output_dir='./outputs', gamma=1.8,
                  scales=[15], k=0.02, print_info=False):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 检查输入路径是文件还是目录
    if os.path.isdir(image_path):
        # 如果是目录，处理目录下所有支持的图像文件
        results = []
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

        # 使用线程池并行处理多个图像
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            futures = []

            for filename in os.listdir(image_path):
                if any(filename.lower().endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(image_path, filename)
                    future = executor.submit(process_single_image, file_path, over_threshold, under_threshold,
                                             output_dir, gamma, scales, k, print_info)
                    futures.append((file_path, future))

            # 收集结果
            for file_path, future in futures:
                result = future.result()
                results.append((file_path, *result))

        return results
    else:
        # 如果是单个文件，直接处理
        return process_single_image(image_path, over_threshold, under_threshold, output_dir,
                                    gamma, scales, k, print_info)


# 修改process_webcam函数以支持GPU加速
def process_webcam(under_threshold=15, over_threshold=180, gamma=1.8, scales=[15, 80], k=0.02):
    """使用GPU加速的摄像头实时处理模式"""
    # 初始化GPU上下文
    gpu_ctx = get_gpu_context()

    # 尝试访问摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法访问摄像头，请检查摄像头是否正常连接")
        return

    print("摄像头已启动，按'q'键退出")

    # 预加载字体和设置常量
    # font_path = 'C:/Windows/Fonts/simhei.ttf'
    # try:
    #     font_small = ImageFont.truetype(font_path, 20)
    #     font_large = ImageFont.truetype(font_path, 24)
    # except IOError:
    #     print("警告: 无法加载指定字体，将使用默认字体")
    font_small = ImageFont.load_default()
    font_large = ImageFont.load_default()

    # 帧率控制变量
    prev_time = time.time()
    fps = 0
    fps_alpha = 0.1  # 平滑因子

    # 增加处理间隔以提高帧率
    process_every_n_frames = 2  # 每2帧处理一次
    display_update_frames = 5  # 每5帧更新一次显示信息
    frame_count = 0
    last_processed_frame = None
    last_display_image = None
    title_img = None

    # 缓存处理结果
    last_under_percentage = 0
    last_over_percentage = 0
    frame_cache = {}  # 用于缓存处理过的帧的哈希值

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧，退出...")
            break

        # 计算当前帧率
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        # 平滑帧率显示
        if dt > 0:
            current_fps = 1.0 / dt
            fps = fps * (1 - fps_alpha) + current_fps * fps_alpha

        # 使用帧缓存机制，避免重复处理类似的帧
        frame_hash = None
        if frame_count % process_every_n_frames == 0:
            # 仅在需要处理时计算帧哈希
            small_hash_frame = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_NEAREST)
            frame_hash = hash(small_hash_frame.tobytes())

            # 检查是否在缓存中
            if frame_hash in frame_cache:
                processed_frame, last_under_percentage, last_over_percentage = frame_cache[frame_hash]
                last_processed_frame = processed_frame
            else:
                # 降低处理分辨率以提高速度
                h, w = frame.shape[:2]
                scale_factor = max(1, min(w, h) / 480)

                if scale_factor > 1.1:
                    proc_w = int(w / scale_factor)
                    proc_h = int(h / scale_factor)
                    small_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                else:
                    small_frame = frame

                # 检测曝光情况
                under_percentage, over_percentage = fast_check_exposure(small_frame, under_threshold, over_threshold)
                last_under_percentage = under_percentage
                last_over_percentage = over_percentage

                # 处理图像
                processed_frame = small_frame.copy()

                if under_percentage - over_percentage > 0.1:
                    # 对欠曝区域应用GPU低光照增强
                    gamma_value = min(2.0, 1.1 + under_percentage / 100)
                    processed_frame = adjust_gamma(processed_frame, gamma=gamma_value)
                elif over_percentage - under_percentage > 0.1:
                    # 对过曝区域处理
                    gamma_value = max(0.4, 0.8 - over_percentage / 200)
                    processed_frame = adjust_gamma(processed_frame, gamma=gamma_value)

                    # 仅在严重过曝时使用SSR算法
                    if over_percentage > 90 and frame_count % (process_every_n_frames * 5) == 0:
                        processed_frame = fast_SSR(processed_frame, scale=scales[0], k=k)

                # 如果之前降低了分辨率，现在恢复原始大小
                if scale_factor > 1.1:
                    processed_frame = cv2.resize(processed_frame, (w, h), interpolation=cv2.INTER_LINEAR)

                # 保存到缓存
                if len(frame_cache) > 20:  # 限制缓存大小
                    # 移除最早的项
                    frame_cache.pop(next(iter(frame_cache)))
                frame_cache[frame_hash] = (processed_frame, under_percentage, over_percentage)

                last_processed_frame = processed_frame
        else:
            # 使用上一帧处理结果
            processed_frame = last_processed_frame if last_processed_frame is not None else frame

        # 减少UI更新频率，大幅提高性能
        show_ui = False
        if frame_count % display_update_frames == 0 or last_display_image is None:
            show_ui = True
        elif last_display_image is not None:
            # 重用上一帧的显示图像
            cv2.imshow('Camera Output', last_display_image)

        # 只在需要更新UI时执行
        if show_ui:
            # 原始帧添加文本
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 30), f"FPS:{fps:.1f} under: {last_under_percentage:.1f}% | over: {last_over_percentage:.1f}%",
                      font=font_small, fill=(255, 0, 0))
            frame_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # 处理后帧添加文本
            img_pil = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 30), f"under: {last_under_percentage:.1f}% | over: {last_over_percentage:.1f}%",
                      font=font_small, fill=(0, 255, 0))

            # 添加帧率显示
            fps_text = f"FPS:{fps:.1f} GPU"
            text_width = draw.textlength(fps_text, font=font_small)
            draw.text((processed_frame.shape[1] - text_width - 10, 30), fps_text,
                      font=font_small, fill=(0, 255, 0))

            processed_frame_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # 横向拼接两个图像
            display_image = np.hstack((frame_with_text, processed_frame_with_text))

            # 使用预渲染的标题
            if frame_count == 0 or title_img is None:
                window_title_img = np.zeros((50, display_image.shape[1], 3), dtype=np.uint8)
                img_pil = Image.fromarray(window_title_img)
                draw = ImageDraw.Draw(img_pil)

                try:
                    text = 'before(left)  | after(right)'
                    text_width = draw.textlength(text, font=font_large)
                    center_x = display_image.shape[1] // 2 - text_width // 2
                    draw.text((center_x, 10), text, font=font_large, fill=(255, 255, 255))
                    title_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    # 如果使用字体失败，使用简单文字
                    cv2.putText(window_title_img, "Original (Left) | Processed (Right)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    title_img = window_title_img

            # 将标题图像和显示图像垂直拼接
            display_with_title = np.vstack((title_img, display_image))

            # 保存当前显示图像以便重用
            last_display_image = display_with_title
            cv2.imshow('Camera Output', display_with_title)

        # 检测按键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    # 清理GPU资源
    if gpu_ctx:
        gpu_ctx.cleanup()


# 添加命令行参数解析
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='自适应光照算法 (GPU加速版)')
    parser.add_argument('--webcam', action='store_true', help='使用摄像头实时处理')
    parser.add_argument('--input', default='./inputs', help='输入图像或目录路径')
    parser.add_argument('--output', default='./outputs', help='输出目录路径')
    parser.add_argument('--under', type=float, default=30, help='欠曝光阈值')
    parser.add_argument('--over', type=float, default=220, help='过曝光阈值')
    parser.add_argument('--gamma', type=float, default=1.8, help='伽马校正参数')
    parser.add_argument('--gpu', action='store_true', default=True, help='使用GPU加速 (ARM Mali-G610 MP4)')

    args = parser.parse_args()

    # 初始化GPU上下文
    if args.gpu:
        gpu_ctx = get_gpu_context()
        if not gpu_ctx or not gpu_ctx.initialized:
            print("警告: 无法初始化GPU，将使用CPU处理")

    if args.webcam:
        # 使用摄像头模式
        process_webcam(under_threshold=args.under, over_threshold=args.over, gamma=args.gamma)
    else:
        # 使用文件处理模式
        process_image(args.input, output_dir=args.output,
                      under_threshold=args.under, over_threshold=args.over,
                      gamma=args.gamma, print_info=False)
