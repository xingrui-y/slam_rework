#include "point_struct.h"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#define SEARCH_RADIUS 5

class RgbdKeyPointStruct::RgbdKeyPointStructImpl
{
  public:
    RgbdKeyPointStructImpl();

    cv::Mat image;
    std::vector<Eigen::Vector3f> key_points;
    std::vector<cv::KeyPoint> cv_keypoints;
    std::vector<int> correspondence;

    static cv::Ptr<cv::FastFeatureDetector> fast_detector;
    static cv::Ptr<cv::BRISK> brisk_detector;
    static cv::Ptr<cv::xfeatures2d::SURF> surf_detector;
};

cv::Ptr<cv::FastFeatureDetector> RgbdKeyPointStruct::RgbdKeyPointStructImpl::fast_detector;
cv::Ptr<cv::BRISK> RgbdKeyPointStruct::RgbdKeyPointStructImpl::brisk_detector;
cv::Ptr<cv::xfeatures2d::SURF> RgbdKeyPointStruct::RgbdKeyPointStructImpl::surf_detector;

RgbdKeyPointStruct::RgbdKeyPointStructImpl::RgbdKeyPointStructImpl()
{
    if (!fast_detector)
        fast_detector = cv::FastFeatureDetector::create();
    if (!brisk_detector)
        brisk_detector = cv::BRISK::create();
    if (!surf_detector)
        surf_detector = cv::xfeatures2d::SURF::create();
}

void RgbdKeyPointStruct::detect(const cv::Mat image, const cv::Mat depth, const IntrinsicMatrix K)
{
    impl->image = image.clone();
    std::vector<cv::KeyPoint> key_points;
    impl->fast_detector->detect(image, key_points);

    for (int i = 0; i < key_points.size(); ++i)
    {
        auto &key = key_points[i];
        int x = (int)(key.pt.x + 0.5f);
        int y = (int)(key.pt.y + 0.5f);
        float z = depth.at<float>(y, x);
        if (z == z)
        {
            Eigen::Vector3f point;
            point(2) = z;
            point(0) = K.invfx * (key.pt.x - K.cx) * z;
            point(1) = K.invfy * (key.pt.y - K.cy) * z;
            impl->key_points.push_back(point);
            impl->cv_keypoints.push_back(key);
        }
    }
}

void RgbdKeyPointStruct::clear_struct()
{
    impl->key_points.clear();
    impl->cv_keypoints.clear();
    impl->correspondence.clear();
}

cv::Mat RgbdKeyPointStruct::get_image() const
{
    return impl->image;
}

cv::Mat RgbdKeyPointStruct::compute_surf() const
{
    cv::Mat desc;
    impl->surf_detector->compute(impl->image, impl->cv_keypoints, desc);
    return desc;
}

cv::Mat RgbdKeyPointStruct::compute_brisk() const
{
    cv::Mat desc;
    impl->brisk_detector->compute(impl->image, impl->cv_keypoints, desc);
    return desc;
}

std::vector<cv::KeyPoint> RgbdKeyPointStruct::get_cv_keypoints() const
{
    return impl->cv_keypoints;
}

std::vector<Eigen::Vector3f> RgbdKeyPointStruct::get_key_points() const
{
    return impl->key_points;
}

int compute_match_score(const cv::Mat ref, const cv::Mat curr)
{
    int score = 0;
    for (int y = 0; y < ref.rows; ++y)
    {
        for (int x = 0; x < ref.cols; ++x)
        {
            score += ref.ptr<uchar>(y, x) - curr.ptr<uchar>(y, x);
        }
    }
    return score;
}

int RgbdKeyPointStruct::count_visible_keypoints(const Sophus::SE3d pose_update, IntrinsicMatrix K) const
{
    int count = 0;
    for (auto key : impl->key_points)
    {
        key = pose_update.cast<float>() * key;
        float x = K.fx * key(0) / key(2) + K.cx;
        float y = K.fy * key(1) / key(2) + K.cy;
        if (x < 0 || y < 0 || x > K.width - 1 || y > K.height - 1)
            continue;

        count++;
    }
    return count;
}

void RgbdKeyPointStruct::match(RgbdKeyPointStructPtr ref, const Sophus::SE3d pose_curr_to_ref, IntrinsicMatrix K)
{
    cv::Mat desc_ref = ref->compute_surf();
    cv::Mat desc_curr = this->compute_surf();
    std::vector<cv::KeyPoint> ref_cv_keypoints = ref->get_cv_keypoints();
    std::vector<Eigen::Vector3f> ref_keypoints = ref->get_key_points();
    int num_ref_keypoints = ref_cv_keypoints.size();
    std::vector<int> corresp_vec(impl->cv_keypoints.size());
    std::vector<cv::DMatch> match_list;

    for (int i = 0; i < impl->cv_keypoints.size(); ++i)
    {
        Eigen::Vector3f key = impl->key_points[i];
        key = pose_curr_to_ref.cast<float>() * key;
        float x = K.fx * key(0) / key(2) + K.cx;
        float y = K.fy * key(1) / key(2) + K.cy;
        if (x < 0 || y < 0 || x > K.width - 1 || y > K.height - 1)
            continue;

        int u = (int)(x + 0.5);
        int v = (int)(y + 0.5);
        int x0 = std::max(0, u - SEARCH_RADIUS);
        int x1 = std::min((int)(K.width - 1), u + SEARCH_RADIUS);
        int y0 = std::max(0, v - SEARCH_RADIUS);
        int y1 = std::min((int)(K.height - 1), v + SEARCH_RADIUS);

        cv::Mat desc_i = desc_curr.row(i);
        double min_score = 64;
        int best_match = -1;
        for (int j = 0; j < num_ref_keypoints; ++j)
        {
            cv::KeyPoint &key_j = ref_cv_keypoints[j];
            Eigen::Vector3f &key_eigen = ref_keypoints[j];
            if (key_j.pt.x < x0 || key_j.pt.x > x1 || key_j.pt.y < y0 || key_j.pt.y > y1 || (key - key_eigen).norm() > 0.1)
                continue;

            cv::Mat desc_j = desc_ref.row(j);
            double score = cv::norm(desc_i, desc_j, cv::NORM_L2);
            if (score < min_score)
            {
                min_score = score;
                best_match = j;
            }
        }

        corresp_vec[i] = best_match;
        cv::DMatch match;
        match.queryIdx = i;
        match.trainIdx = best_match;

        if (best_match >= 0)
            match_list.push_back(match);
    }

    cv::Mat curr_image = this->get_image();
    auto curr_cv_keypoints = this->get_cv_keypoints();
    cv::Mat ref_image = ref->get_image();
    cv::Mat outImg;
    cv::drawMatches(curr_image, curr_cv_keypoints, ref_image, ref_cv_keypoints, match_list, outImg);

    cv::imshow("outImg", outImg);
    cv::waitKey(1);
}

RgbdKeyPointStruct::RgbdKeyPointStruct() : impl(new RgbdKeyPointStructImpl())
{
}