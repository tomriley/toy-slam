import cv2 as cv
import numpy as np

np.set_printoptions(suppress = True) # supress e notation

def build_4_matrix(rot, t):
    mat = np.zeros((4,4), dtype=np.float)
    mat[:3, :3] = rot
    mat[0][3] = t[0]
    mat[1][3] = t[1]
    mat[2][3] = t[2]
    mat[3][3] = 1.0
    return mat

# class Point:
#     def __init__(self, screen, world):
#         self.screen = screen
#         self.world = world

class Frame:
    def __init__(self, image):
        self.image = image
        orb = cv.ORB_create() # FIXME reuse?
        self.kp, self.des = orb.detectAndCompute(image, None)
        self.points = []
        self.lines = None
        self.pose = np.eye(4, 4, dtype=float)

class SLAM:
    def __init__(self, K, P):
        #self.prev_frame = None
        self.frames = []
        self.kp = None
        self.des = None
        self.K = K
        self.P = P
        self.K_inv = np.linalg.inv(K)
        self.colors = np.random.randint(0, 255, (200, 3))
    
    # def find_features(self, grey_frame):
    #     feature_params = dict(maxCorners = 300,
    #                         qualityLevel = 0.3,
    #                         minDistance = grey_frame.shape[0] // 10,
    #                         blockSize = 15)
    #     self.p0 = cv.goodFeaturesToTrack(grey_frame, mask=None, **feature_params)
    #     self.prev_frame = grey_fram

    def process_image(self, image):
        frame = Frame(image)
        self.frames.append(frame)
        
        # do slam if we more than one frame
        if len(self.frames) >= 2:
            self.match(self.frames[-2], frame)
        
        return frame
    
    def match(self, f0, f1):
        pts0 = []
        pts1 = []
        lines = np.zeros((f0.image.shape[0], f0.image.shape[1], 3), dtype='uint8')
        
        bf = cv.BFMatcher(cv.NORM_HAMMING)
        matches = bf.knnMatch(f0.des, f1.des, k=2) # best 2 matches
        # matches = sorted(matches, key = lambda x:x.distance) should aready be sorted
        
        # Build list of matching image coordinates
        for i, (m0, m1) in enumerate(matches):
            if True or m0.distance < 30: # good match
                if True or m1.distance > m0.distance * 1.7: # highly specific
                    pt0 = f0.kp[m0.queryIdx].pt
                    pt1 = f1.kp[m0.trainIdx].pt
                    pt0i = (int(pt0[0]), int(pt0[1]))
                    pt1i = (int(pt1[0]), int(pt1[1]))
                    pts0.append(pt0i)
                    pts1.append(pt1i)
                    lines = cv.line(lines, pt0i, pt1i, self.colors[i % len(self.colors)].tolist(), 2)

        pts0 = np.int32(pts0)
        pts1 = np.int32(pts1)
        
        F, mask = cv.findFundamentalMat(pts0, pts1, cv.FM_LMEDS) # cv.FM_RANSAC
        inliers0 = []
        inliers1 = []
        for i in range(len(mask)):
            if mask[i]:
                #pt0 = [pts0[i][0], pts0[i][1], 1.0]
                #pt1 = [pts1[i][0], pts1[i][1], 1.0]
                #inliers0.append(self.K_inv.dot(pt0))
                #inliers1.append(self.K_inv.dot(pt1))
                inliers0.append(pts0[i])
                inliers1.append(pts1[i])
        
        inliers0 = np.float32(inliers0)
        inliers1 = np.float32(inliers1)

        # convert F to essential matrix
        # TODO check on just using opencv findEssentialMat
        E = self.K.T.dot(F).dot(self.K)

        print("E=", E)


        E, mask = cv.findEssentialMat(pts0, pts1, self.K) # cv.FM_RANSAC
        inliers0 = []
        inliers1 = []
        for i in range(len(mask)):
            if mask[i]:
                inliers0.append(pts0[i])
                inliers1.append(pts1[i])
        
        inliers0 = np.float32(inliers0)
        inliers1 = np.float32(inliers1)

        print("E2=", E)


        # decompose into the 4 possible solutions
        R1, R2, t = cv.decomposeEssentialMat(E)
        t = t.reshape(-1)

        Kh = np.concatenate((self.K, np.zeros((3,1), dtype=float)), axis=1)
        #print("Kh", Kh)

        trans = [
            build_4_matrix(R1, t),
            build_4_matrix(R2, t),
            build_4_matrix(R1, -t),
            build_4_matrix(R2, -t)
        ]
        projs = [Kh @ t for t in trans]

        scores = []
        f1rel_positions = []

        # Find camera poses that result in highest number of points in front of cameras
        for i in range(4):
            points0 = cv.triangulatePoints(Kh, projs[i], inliers0.T, inliers1.T)
            # transform to other camera
            points1 = trans[i] @ points0
            
            # unhomogenize
            points0 = points0[:3, :] / points0[3, :]
            points1 = points1[:3, :] / points1[3, :]

            valid_f1_pts = []
            score = 0
            for i in range(len(points0[0])):
                if points0[2][i] > 0 and points1[2][i] > 0:
                    score += 1
                    valid_f1_pts.append([points1[0][i], points1[1][i], points1[2][i], 1.0])

            f1rel_positions.append(np.array(valid_f1_pts))

            score = sum(points0[2, :] > 0) + sum(points1[2, :] > 0)
            scores.append(score)

        print("scores=", scores)
        idx = np.argmax(scores)

        f1.pose = f0.pose @ np.linalg.inv(trans[idx])
        print(f1rel_positions[idx].shape)
        f1_pts = f1rel_positions[idx].T
        f1_pts = (f1.pose @ f1_pts).T
        #f1_pts = f1_pts.T

        print("world_pt=", f1_pts[0])
        f1.points = f1_pts

        #cv.drawKeypoints(grey_frame, kp1, out, color=[0,0,255])
        f1.lines = lines

        