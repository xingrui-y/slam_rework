#include "point_struct.h"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

class KeyPointStruct::KeyPointStructImpl
{
  public:
    KeyPointStructImpl();
    cv::Mat compute();
    void detect(const RgbdFramePtr frame, const IntrinsicMatrix K);
    void project_and_show(const RgbdFramePtr frame, const Sophus::SE3d pose, const IntrinsicMatrix K);
    int match(KeyPointStructPtr reference_struct, Sophus::SE3d pose_to_ref, IntrinsicMatrix K);
    std::vector<cv::KeyPoint> get_key_points_cv() const;

    static cv::Ptr<cv::FastFeatureDetector> fast_detector_;
    static cv::Ptr<cv::xfeatures2d::SURF> surf_detector_;

    cv::Mat image_;
    std::vector<Point3d> key_points_3d_;
    std::vector<cv::KeyPoint> key_points_cv_;
};

cv::Ptr<cv::FastFeatureDetector> KeyPointStruct::KeyPointStructImpl::fast_detector_;
cv::Ptr<cv::xfeatures2d::SURF> KeyPointStruct::KeyPointStructImpl::surf_detector_;

KeyPointStruct::KeyPointStructImpl::KeyPointStructImpl()
{
    if (!fast_detector_)
        fast_detector_ = cv::FastFeatureDetector::create(25);
    if (!surf_detector_)
        surf_detector_ = cv::xfeatures2d::SURF::create();
}

void KeyPointStruct::KeyPointStructImpl::detect(const RgbdFramePtr frame, const IntrinsicMatrix K)
{
    cv::Mat image = frame->get_image();
    cv::Mat depth = frame->get_depth();
    image_ = image.clone();
    std::vector<cv::KeyPoint> temp_key_points;
    fast_detector_->detect(image, temp_key_points);
    int num_keypoints = temp_key_points.size();

    for (int i = 0; i < num_keypoints; ++i)
    {
        auto &kp = temp_key_points[i];
        int x = (int)(kp.pt.x + 0.5);
        int y = (int)(kp.pt.y + 0.5);
        float z = depth.ptr<float>(y)[x];

        if (z == z)
        {
            Point3d pt3d;
            pt3d.x_ = kp.pt.x;
            pt3d.y_ = kp.pt.y;
            pt3d.pos_ = Eigen::Vector3f(z * (x - K.cx) * K.invfy, z * (y - K.cy) * K.invfy, z);
            key_points_3d_.push_back(pt3d);
        }
    }
}

void KeyPointStruct::KeyPointStructImpl::project_and_show(const RgbdFramePtr frame, const Sophus::SE3d keyframe_pose, const IntrinsicMatrix K)
{
    cv::Mat image = frame->get_image();
    Sophus::SE3d pose_from_ref = frame->get_pose().inverse() * keyframe_pose;
    std::vector<cv::KeyPoint> frame_keypoints;
    for (int i = 0; i < key_points_3d_.size(); ++i)
    {
        auto kp = key_points_3d_[i];
        auto pos = pose_from_ref.cast<float>() * kp.pos_;
        float x = K.fx * pos(0) / pos(2) + K.cx;
        float y = K.fy * pos(1) / pos(2) + K.cy;
        if (x < 0 || y < 0 || x > K.width - 1 || y > K.height - 1)
            continue;

        cv::KeyPoint pt;
        pt.pt.x = x;
        pt.pt.y = y;
        frame_keypoints.push_back(pt);
    }

    cv::Mat outImg;
    cv::drawKeypoints(image, frame_keypoints, outImg, cv::Scalar(0, 255, 0));
    cv::imshow("outImg", outImg);
    cv::waitKey(1);
}

std::vector<cv::KeyPoint> KeyPointStruct::KeyPointStructImpl::get_key_points_cv() const
{
    std::vector<cv::KeyPoint> key_points;
    std::transform(key_points_3d_.begin(), key_points_3d_.end(), std::back_inserter(key_points), [](Point3d pt) -> cv::KeyPoint {cv::KeyPoint kp; kp.pt.x = pt.x_; kp.pt.y = pt.y_; return kp; });
    return key_points;
}

int KeyPointStruct::KeyPointStructImpl::match(KeyPointStructPtr reference_struct, Sophus::SE3d pose_to_ref, IntrinsicMatrix K)
{
    cv::Mat desc_ref = reference_struct->compute();
    cv::Mat desc_curr = this->compute();
    auto kp_ref = reference_struct->get_key_points_3d();
    std::vector<cv::DMatch> match_list;
    for (int i = 0; i < key_points_3d_.size(); ++i)
    {
        Eigen::Vector3f pt = pose_to_ref.cast<float>() * key_points_3d_[i].pos_;
        float x = K.fx * pt(0) / pt(2) + K.cx;
        float y = K.fy * pt(1) / pt(2) + K.cy;
        if (x < 0 || y < 0 || x > K.width - 1 || y > K.height - 1)
            continue;

        int u = (int)(x + 0.5);
        int v = (int)(y + 0.5);
        int x0 = std::max(0, u - 5);
        int x1 = std::min((int)(K.width - 1), u + 5);
        int y0 = std::max(0, v - 5);
        int y1 = std::min((int)(K.height - 1), v + 5);

        cv::Mat desc_i = desc_curr.row(i);
        double min_score = std::numeric_limits<double>::max();
        int best_match = -1;
        for (int j = 0; j < kp_ref.size(); ++j)
        {
            float ref_x = kp_ref[j].x_;
            float ref_y = kp_ref[j].y_;
            if (ref_x < x0 || ref_x > x1 || ref_y < y0 || ref_y > y1)
                continue;

            cv::Mat desc_j = desc_ref.row(j);
            double score = cv::norm(desc_i, desc_j, cv::NORM_L2);
            if (score < min_score)
            {
                min_score = score;
                best_match = j;
            }
        }

        cv::DMatch match;
        match.queryIdx = i;
        match.trainIdx = best_match;

        if (best_match >= 0)
            match_list.push_back(match);
    }

    cv::Mat ref_image = reference_struct->get_image();
    auto curr_cv_keypoints = this->get_key_points_cv();
    auto ref_cv_keypoints = reference_struct->get_key_points_cv();
    cv::Mat outImg;
    cv::drawMatches(image_, curr_cv_keypoints, ref_image, ref_cv_keypoints, match_list, outImg);

    cv::imshow("outImg", outImg);
    cv::waitKey(1);

    return match_list.size();
}

cv::Mat KeyPointStruct::KeyPointStructImpl::compute()
{
    cv::Mat desc;
    auto key_points = this->get_key_points_cv();
    surf_detector_->compute(image_, key_points, desc);
    return desc;
}

KeyPointStruct::KeyPointStruct() : impl(new KeyPointStructImpl())
{
}

void KeyPointStruct::detect(const RgbdFramePtr frame, const IntrinsicMatrix K)
{
    impl->detect(frame, K);
}

int KeyPointStruct::match(KeyPointStructPtr reference_struct, Sophus::SE3d pose_to_ref, IntrinsicMatrix K)
{
    return impl->match(reference_struct, pose_to_ref, K);
}

void KeyPointStruct::project_and_show(const RgbdFramePtr frame, const Sophus::SE3d pose, const IntrinsicMatrix K)
{
    impl->project_and_show(frame, pose, K);
}

cv::Mat KeyPointStruct::compute()
{
    return impl->compute();
}

std::vector<cv::KeyPoint> KeyPointStruct::get_key_points_cv() const
{
    return impl->get_key_points_cv();
}

std::vector<Point3d> KeyPointStruct::get_key_points_3d() const
{
    return impl->key_points_3d_;
}

cv::Mat KeyPointStruct::get_image() const
{
    return impl->image_;
}