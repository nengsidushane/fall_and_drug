import cv2
import numpy as np
from affine_ransac import Ransac
from affine_transform import Affine
from skimage.metrics import structural_similarity as ssim


# The ration of the best match over second best match
#      distance of best match
# ------------------------------- <= MATCH_RATIO
#  distance of second best match
RATIO = 0.8


class Align():

    def __init__(self, source, target, K=3, threshold=1):
        ''' __INIT__

            Initialize the instance.

            Input arguments:

            - source_path : the path of sorce image that to be warped
            - target_path : the path of target image
            - K : the number of corresponding points, default is 3
            - threshold : a threshold determins which points are outliers
            in the RANSAC process, if the residual is larger than threshold,
            it can be regarded as outliers, default value is 1

        '''
        """支持直接传入numpy数组"""
        if isinstance(source, np.ndarray):
            self.source = source
        else:
            self.source = cv2.imread(source)
        if isinstance(target, np.ndarray):
            self.target = target
        else:
            self.target = cv2.imread(target)

        self.K = K
        self.threshold = threshold

    def read_image(self, path, mode=1):
        ''' READ_IMAGE

            Load image from file path.

            Input arguments:

            - path : the image to be read
            - mode : 1 for reading color image, 0 for grayscale image
            default is 1

            Output:

            - the image to be processed

        '''

        return cv2.imread(path, mode)

    def extract_SIFT(self, img):
        ''' EXTRACT_SIFT

            Extract SIFT descriptors from the given image.

            Input argument:

            - img : the image to be processed

            Output:

            -kp : positions of key points where descriptors are extracted
            - desc : all SIFT descriptors of the image, its dimension
            will be n by 128 where n is the number of key points


        '''

        # Convert the image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract key points and SIFT descriptors
        sift = cv2.SIFT_create()
        kp, desc = sift.detectAndCompute(img_gray, None)

        # Extract positions of key points
        kp = np.array([p.pt for p in kp]).T

        return kp, desc

    def match_SIFT(self, desc_s, desc_t):
        ''' MATCH_SIFT

            Match SIFT descriptors of source image and target image.
            Obtain the index of conrresponding points to do estimation
            of affine transformation.

            Input arguments:

            - desc_s : descriptors of source image
            - desc_t : descriptors of target image

            Output:

            - fit_pos : index of corresponding points

        '''

        # Match descriptor and obtain two best matches
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_s, desc_t, k=2)

        # Initialize output variable
        fit_pos = np.array([], dtype=np.int32).reshape((0, 2))

        matches_num = len(matches)
        for i in range(matches_num):
            # Obtain the good match if the ration id smaller than 0.8
            if matches[i][0].distance <= RATIO * matches[i][1].distance:
                temp = np.array([matches[i][0].queryIdx,
                                 matches[i][0].trainIdx])
                # Put points index of good match
                fit_pos = np.vstack((fit_pos, temp))

        return fit_pos

    def affine_matrix(self, kp_s, kp_t, fit_pos):
        ''' AFFINE_MATRIX

            Compute affine transformation matrix by corresponding points.

            Input arguments:

            - kp_s : key points from source image
            - kp_t : key points from target image
            - fit_pos : index of corresponding points

            Output:

            - M : the affine transformation matrix whose dimension
            is 2 by 3

        '''

        # Extract corresponding points from all key points
        kp_s = kp_s[:, fit_pos[:, 0]]
        kp_t = kp_t[:, fit_pos[:, 1]]

        # Apply RANSAC to find most inliers
        _, _, inliers = Ransac(self.K, self.threshold).ransac_fit(kp_s, kp_t)

        # Extract all inliers from all key points
        kp_s = kp_s[:, inliers[0]]
        kp_t = kp_t[:, inliers[0]]

        # Use all inliers to estimate transform matrix
        A, t = Affine().estimate_affine(kp_s, kp_t)
        M = np.hstack((A, t))

        return M

    def warp_image(self, source, target, M):
        ''' WARP_IMAGE

            Warp the source image into target with the affine
            transformation matrix.

            Input arguments:

            - source : the source image to be warped
            - target : the target image
            - M : the affine transformation matrix

        '''

        # Obtain the size of target image
        rows, cols, _ = target.shape

        # Warp the source image
        warp = cv2.warpAffine(source, M, (cols, rows))

        # Merge warped image with target image to display
        merge = np.uint8(target * 0.5 + warp * 0.5)

        # Show the result
        cv2.imshow('img', merge)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return

    def compute_similarity(self, img1=None, img2=None, use_ssim=True, weight_ssim=0.6):
        ''' COMPUTE_SIMILARITY

            计算两张图片间的相似度，基于SIFT特征匹配和SSIM

            输入参数:
            - img1_path: 第一张图片路径，如果为None则使用source_path
            - img2_path: 第二张图片路径，如果为None则使用target_path
            - use_ssim: 是否使用SSIM增强相似度计算
            - weight_ssim: SSIM权重(0-1)，SIFT特征匹配权重为(1-weight_ssim)

            输出:
            - similarity: 相似度分数 (0-1)，1表示完全匹配
            - match_info: 包含匹配信息的字典
        '''
        # 如果未提供路径，使用实例化时的路径
        if img1 is None:
            img1 = self.source
        if img2 is None:
            img2 = self.target
        
        # 提取SIFT特征
        kp1, desc1 = self.extract_SIFT(img1)
        kp2, desc2 = self.extract_SIFT(img2)
        
        # 匹配SIFT特征
        fit_pos = self.match_SIFT(desc1, desc2)
        
        # 计算SIFT相似度分数
        # 1. 基于匹配点的数量
        matches_count = fit_pos.shape[0]
        features_count = min(desc1.shape[0], desc2.shape[0])
        
        # 防止除以0
        if features_count == 0:
            ratio_score = 0
        else:
            ratio_score = matches_count / features_count
            
        # 2. 如果有匹配点，计算匹配质量
        if matches_count > 0:
            # 提取匹配点
            pts1 = kp1[:, fit_pos[:, 0]]
            pts2 = kp2[:, fit_pos[:, 1]]
            
            # 通过RANSAC找到内点
            _, _, inliers = Ransac(self.K, self.threshold).ransac_fit(pts1, pts2)
            
            if inliers is not None:
                inliers_count = len(inliers[0])
                inlier_ratio = inliers_count / matches_count
            else:
                inlier_ratio = 0
                
            # 综合匹配数量和质量计算SIFT相似度分数
            sift_similarity = 0.5 * ratio_score + 0.5 * inlier_ratio
        else:
            sift_similarity = 0
            inlier_ratio = 0
        
        # 计算最终相似度分数
        if use_ssim:
            # 计算SSIM相似度
            # 将图像转换为灰度
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # 确保两个图像尺寸相同
            if gray1.shape != gray2.shape:
                # 将第二个图像调整为与第一个图像相同的尺寸
                gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
            
            # 计算SSIM值
            ssim_score = ssim(gray1, gray2)
            
            # 结合SIFT和SSIM得分
            similarity = weight_ssim * ssim_score + (1 - weight_ssim) * sift_similarity
        else:
            similarity = sift_similarity
        
        # 准备返回的匹配信息
        match_info = {
            'total_features1': desc1.shape[0],
            'total_features2': desc2.shape[0],
            'matches_count': matches_count,
            'inlier_ratio': inlier_ratio,
            'sift_similarity': sift_similarity
        }
        
        # 如果使用了SSIM，添加相关信息
        if use_ssim:
            match_info['ssim_score'] = ssim_score
            match_info['ssim_weight'] = weight_ssim
        
        match_info['similarity_score'] = similarity
        
        return similarity, match_info

    def align_image(self):
        ''' ALIGN_IMAGE

            Warp the source image into target image.
            Two images' path are provided when the
            instance Align() is created.

        '''

        # Load source image and target image
        img_source = self.read_image(self.source_path)
        img_target = self.read_image(self.target_path)

        # Extract key points and SIFT descriptors from
        # source image and target image respectively
        kp_s, desc_s = self.extract_SIFT(img_source)
        kp_t, desc_t = self.extract_SIFT(img_target)

        # Obtain the index of corresponding points
        fit_pos = self.match_SIFT(desc_s, desc_t)

        # Compute the affine transformation matrix
        M = self.affine_matrix(kp_s, kp_t, fit_pos)

        # Warp the source image and display result
        self.warp_image(img_source, img_target, M)

        return
