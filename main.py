import numpy as np
import cv2


def track_feature(curr_img, prev_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    # Use SIFT to detect and compute keypoints and descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(prev_gray, None)
    kp2, des2 = sift.detectAndCompute(img_gray, None)

    # Use BFMatcher to perform feature matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    # Get matched keypoints
    matched_pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    matched_pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calculate optical flow on the matched points
    _, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, img_gray, matched_pts1, None)

    # Filter out points that were not successfully tracked
    good_matched_pts1 = matched_pts1[status.ravel() == 1]
    good_matched_pts2 = matched_pts2[status.ravel() == 1]

    # Draw the tracks
    for pt1, pt2 in zip(good_matched_pts1, good_matched_pts2):
        a, b = pt1.ravel()
        c, d = pt2.ravel()
        cv2.line(prev_img, (int(a), int(b)),
                 (int(c), int(d)), (0, 200, 20), 2)
        cv2.circle(prev_img, (int(c), int(d)), 3, (100, 25, 200), -1)

    return prev_img


def main():
    PATH = 'C:/00/image_0/{0:06d}.png'
    # PATH = "vid.mp4" # or camera id
    # cap = cv2.VideoCapture(PATH)

    # _, prev = cap.read()
    prev = cv2.imread(PATH.format(0))

    for i in range(1, 3000):
        # _, curr = cap.read()
        curr = cv2.imread(PATH.format(i))

        cv2.imshow('frame', track_feature(curr, prev))
        prev = curr
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
