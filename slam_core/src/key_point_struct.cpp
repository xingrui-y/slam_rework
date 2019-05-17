#include "key_point_struct.h"
#include "stop_watch.h"
#include "opencv_recorder.h"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

class KeyPointStruct::KeyPointStructImpl
{
public:
    KeyPointStructImpl();
    cv::Mat compute();
    int detect(const RgbdFramePtr frame);
    int match_by_descriptors(KeyPointStructPtr current_struct);
    int match_by_pose_constraint(KeyPointStructPtr current_struct, IntrinsicMatrix K, bool count_only);
    int match_by_pose_constraint(KeyPointStructPtr current_struct, IntrinsicMatrix K, bool count_only, const char *window_name);
    int create_points(int maximum_number, const IntrinsicMatrix K);
    void track_key_points(const KeyPointStructPtr current, const IntrinsicMatrix K);
    std::vector<cv::KeyPoint> get_key_points_cv() const;

    cv::Mat image_, desc_;
    bool has_descriptor_;

    RgbdFramePtr reference_frame_;
    std::vector<KeyPoint> key_points_;
    size_t num_key_points_;

    static cv::Ptr<cv::FastFeatureDetector> fast_detector_;
    static cv::Ptr<cv::xfeatures2d::SURF> surf_detector_;
    static cv::Ptr<cv::xfeatures2d::HarrisLaplaceFeatureDetector> harris_detector_;
    static cv::Ptr<cv::DescriptorMatcher> bf_matcher_;
};

cv::Ptr<cv::FastFeatureDetector> KeyPointStruct::KeyPointStructImpl::fast_detector_;
cv::Ptr<cv::xfeatures2d::SURF> KeyPointStruct::KeyPointStructImpl::surf_detector_;
cv::Ptr<cv::xfeatures2d::HarrisLaplaceFeatureDetector> KeyPointStruct::KeyPointStructImpl::harris_detector_;
cv::Ptr<cv::DescriptorMatcher> KeyPointStruct::KeyPointStructImpl::bf_matcher_;

KeyPointStruct::KeyPointStructImpl::KeyPointStructImpl()
{
    if (!fast_detector_)
        fast_detector_ = cv::FastFeatureDetector::create(25);
    if (!surf_detector_)
        surf_detector_ = cv::xfeatures2d::SURF::create(300);
    if (!harris_detector_)
        harris_detector_ = cv::xfeatures2d::HarrisLaplaceFeatureDetector::create();
    if (!bf_matcher_)
        bf_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_SL2);
}

int KeyPointStruct::KeyPointStructImpl::create_points(int maximum_number, const IntrinsicMatrix K)
{
    size_t max_iteration;
    if (maximum_number < 0)
        max_iteration = key_points_.size();
    else
        max_iteration = std::min((size_t)maximum_number, key_points_.size());

    auto frame_pose = reference_frame_->get_pose();
    int num_keypoints_created = 0;
    const size_t frame_id = reference_frame_->get_id();

    for (int i = 0; i < max_iteration; ++i)
    {
        KeyPoint &kp = key_points_[i];

        if (kp.pt3d_ != nullptr)
            continue;

        kp.pt3d_ = std::make_shared<Point3d>();

        Eigen::Vector3d pos_3d;
        pos_3d << kp.z_ * K.invfx * (kp.kp_.pt.x - K.cx), kp.z_ * K.invfy * (kp.kp_.pt.y - K.cy), kp.z_;
        kp.pt3d_->pos_ = frame_pose * pos_3d;
        kp.pt3d_->observations_[frame_id] = i;
        kp.pt3d_->reference_frame_ = reference_frame_;
        num_keypoints_created++;
    }

    return num_keypoints_created;
}

void KeyPointStruct::KeyPointStructImpl::track_key_points(const KeyPointStructPtr current, const IntrinsicMatrix K)
{
    auto current_frame = current->get_reference_frame();
    auto current_pose = current_frame->get_pose();
    auto inv_current_pose = current_pose.inverse();

    auto iter = key_points_.begin();
    auto lend = key_points_.cend();
    auto current_image = current->get_image().clone();

    for (; iter != lend; ++iter)
    {
        auto &kp = *iter;
        if (kp.pt3d_)
        {
            auto pt_in_current = inv_current_pose * kp.pt3d_->pos_;
            float ux = K.fx * pt_in_current(0) / pt_in_current(2) + K.cx;
            float uy = K.fy * pt_in_current(1) / pt_in_current(2) + K.cy;
            if (ux >= 0 && uy >= 0 && ux < K.width && uy < K.height)
            {
                cv::Point2d pt(ux, uy);
                cv::drawMarker(current_image, pt, cv::Scalar(0, 255, 0), cv::MARKER_SQUARE, 8);
                cv::drawMarker(current_image, pt, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 4);
            }
        }
    }

    cv::cvtColor(current_image, current_image, cv::COLOR_RGB2BGR);
    cv::imshow("kp in current", current_image);
    cv::waitKey(1);
}

int KeyPointStruct::KeyPointStructImpl::detect(const RgbdFramePtr frame)
{
    cv::Mat image = frame->get_image();
    cv::Mat depth = frame->get_depth();

    // original image used to visualize key point matches;
    image_ = image.clone();

    // detect all key points from the image even if they don't have a valid depth;
    std::vector<cv::KeyPoint> raw_keypoints;
    surf_detector_->detect(image, raw_keypoints);
    // fast_detector_->detect(image, raw_keypoints);
    int size_keypoints = raw_keypoints.size();

    for (int idx = 0; idx < size_keypoints; ++idx)
    {
        auto &kp_cv = raw_keypoints[idx];
        const int x = (int)(kp_cv.pt.x + 0.5);
        const int y = (int)(kp_cv.pt.y + 0.5);
        float z = depth.ptr<float>(y)[x];
        cv::Vec3b rgb = image.ptr<cv::Vec3b>(y)[x];

        if (!std::isnan(z) && std::isfinite(z) && z > 1e-1)
        {
            KeyPoint kp;
            kp.kp_ = kp_cv;
            kp.pt3d_ = NULL;
            kp.z_ = z;
            kp.rgb_ = Eigen::Vector3f(rgb(2), rgb(1), rgb(0));
            kp.descriptor_ = cv::Mat();
            key_points_.emplace_back(std::move(kp));
        }
    }

    // sort all keypoints in descending order w.r.t. depth estimation
    std::sort(key_points_.begin(), key_points_.end(), [](const KeyPoint &a, const KeyPoint &b) -> bool { return a.z_ < b.z_; });

    // cv::Mat out_image;
    // cv::drawKeypoints(image_, this->get_key_points_cv(), out_image);
    // cv::imshow("keypoints", out_image);
    // cv::waitKey(0);

    reference_frame_ = frame;
    num_key_points_ = key_points_.size();

    return num_key_points_;
}

std::vector<cv::KeyPoint> KeyPointStruct::KeyPointStructImpl::get_key_points_cv() const
{
    std::vector<cv::KeyPoint> key_points;
    std::transform(key_points_.begin(), key_points_.end(), std::back_inserter(key_points), [](const KeyPoint &pt) -> cv::KeyPoint { return pt.kp_; });
    return key_points;
}

int KeyPointStruct::KeyPointStructImpl::match_by_descriptors(KeyPointStructPtr current_struct)
{
    auto reference_key_points = get_key_points_cv();
    auto current_key_points = current_struct->get_key_points_cv();
    auto num_ref_keypoints = reference_key_points.size();
    auto num_curr_keypoints = current_key_points.size();
    auto reference_descriptors = this->desc_;
    auto current_descriptors = current_struct->compute();

    std::vector<cv::DMatch> matches;
}

int KeyPointStruct::KeyPointStructImpl::match_by_pose_constraint(KeyPointStructPtr current_struct, IntrinsicMatrix K, bool count_only)
{
    return match_by_pose_constraint(current_struct, K, count_only, "out image");
}

int KeyPointStruct::KeyPointStructImpl::match_by_pose_constraint(KeyPointStructPtr current_struct, IntrinsicMatrix K, bool count_only, const char *window_name)
{
    cv::Mat descriptors_current = current_struct->compute();
    cv::Mat descriptors_this = this->compute();

    auto &keypoints_current = current_struct->get_key_points();

    const int block_size = 5;
    auto inv_current_pose = current_struct->get_reference_frame()->get_pose().inverse();

    const int current_frame_id = current_struct->get_reference_frame()->get_id();
    int num_keypoint_matched = 0;

    std::vector<bool> indicator_1vs1(keypoints_current.size());
    std::fill(indicator_1vs1.begin(), indicator_1vs1.end(), false);

    // Only for visualization
    std::vector<cv::DMatch> current_matches;

    for (int idx = 0; idx < key_points_.size(); ++idx)
    {
        auto &key = key_points_[idx];
        // auto pose = reference_frame_->get_pose();

        if (key.pt3d_ == nullptr)
            continue;

        Eigen::Vector3d pt_transformed = inv_current_pose * key.pt3d_->pos_;
        float x_project = K.fx * pt_transformed(0) / pt_transformed(2) + K.cx;
        float y_project = K.fy * pt_transformed(1) / pt_transformed(2) + K.cy;
        if (x_project < 0 || y_project < 0 || x_project > K.width - 1 || y_project > K.height - 1)
            continue;

        int u = (int)(x_project + 0.5);
        int v = (int)(y_project + 0.5);
        int x0 = std::max(0, u - block_size);
        int x1 = std::min((int)(K.width - 1), u + block_size);
        int y0 = std::max(0, v - block_size);
        int y1 = std::min((int)(K.height - 1), v + block_size);

        cv::Mat this_descriptor_line = descriptors_this.row(idx);

        double min_score = 0.3;
        int best_match_ptr = -1;
        KeyPoint *best_match_kp = nullptr;

        for (int current_idx = 0; current_idx < keypoints_current.size(); ++current_idx)
        {
            if (indicator_1vs1[current_idx])
                continue;

            auto current_key = keypoints_current[current_idx];
            if (current_key.kp_.pt.x < x0 || current_key.kp_.pt.y < y0 || current_key.kp_.pt.x > x1 || current_key.kp_.pt.y > y1 || std::abs(current_key.z_ - key.z_) > 0.1)
                continue;

            cv::Mat current_descriptor_line = descriptors_current.row(current_idx);

            double score = cv::norm(this_descriptor_line, current_descriptor_line, cv::NORM_L2);

            if (score < min_score)
            {
                min_score = score;
                best_match_ptr = current_idx;
                best_match_kp = &keypoints_current[current_idx];
            }
        }

        if (best_match_ptr >= 0 && best_match_kp != nullptr)
        {
            if (!count_only && !indicator_1vs1[best_match_ptr])
            {

                key.pt3d_->observations_[current_frame_id] = best_match_ptr;
                best_match_kp->pt3d_ = key.pt3d_;
                indicator_1vs1[best_match_ptr] = true;
            }

            num_keypoint_matched++;

            // Visualization
            if (FLAGS_show_image)
            {
                cv::DMatch match;
                match.queryIdx = idx;
                match.trainIdx = best_match_ptr;
                current_matches.emplace_back(std::move(match));
            }
        }
    }

    // Visualize Mathces
    if (FLAGS_show_image)
    {
        cv::Mat out_image;
        cv::Mat current_image = current_struct->get_image();
        auto keypoints_cv_current = current_struct->get_key_points_cv();
        auto keypoints_cv_this = this->get_key_points_cv();

        if (!image_.empty() && !current_image.empty())
        {
            cv::drawMatches(image_, keypoints_cv_this, current_image, keypoints_cv_current, current_matches, out_image, cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0));
            cv::cvtColor(out_image, out_image, cv::COLOR_RGB2BGR);
            cv::imshow(window_name, out_image);
            cv::waitKey(1);
        }
    }

    return num_keypoint_matched;
}

cv::Mat KeyPointStruct::KeyPointStructImpl::compute()
{
    if (desc_.empty())
    {
        auto key_points = this->get_key_points_cv();
        surf_detector_->compute(image_, key_points, desc_);
    }

    return desc_;
}

KeyPointStruct::KeyPointStruct() : impl(new KeyPointStructImpl())
{
}

int KeyPointStruct::detect(const RgbdFramePtr frame)
{
    return impl->detect(frame);
}

RgbdFramePtr KeyPointStruct::get_reference_frame() const
{
    return impl->reference_frame_;
}

int KeyPointStruct::match_by_descriptors(KeyPointStructPtr current_struct)
{
    return impl->match_by_descriptors(current_struct);
}

int KeyPointStruct::match_by_pose_constraint(KeyPointStructPtr current_struct, IntrinsicMatrix K, bool count_only)
{
    return impl->match_by_pose_constraint(current_struct, K, count_only);
}

int KeyPointStruct::match_by_pose_constraint(KeyPointStructPtr current_struct, IntrinsicMatrix K, bool count_only, const char *window_name)
{
    return impl->match_by_pose_constraint(current_struct, K, count_only, window_name);
}

void KeyPointStruct::track_key_points(const KeyPointStructPtr current, const IntrinsicMatrix K)
{
    impl->track_key_points(current, K);
}

int KeyPointStruct::create_points(int maximum_number, const IntrinsicMatrix K)
{
    return impl->create_points(maximum_number, K);
}

cv::Mat KeyPointStruct::compute()
{
    return impl->compute();
}

std::vector<cv::KeyPoint> KeyPointStruct::get_key_points_cv() const
{
    return impl->get_key_points_cv();
}

std::vector<KeyPoint> &KeyPointStruct::get_key_points()
{
    return impl->key_points_;
}

cv::Mat KeyPointStruct::get_image() const
{
    return impl->image_;
}