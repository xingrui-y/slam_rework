#include "tum_wrapper.h"

int find_closest_index(std::vector<double> list, double time)
{
	int idx = -1;
	double min_val = std::numeric_limits<double>::max();
	for (int i = 0; i < list.size(); ++i)
	{
		double d = std::abs(list[i] - time);
		if (d < min_val)
		{
			idx = i;
			min_val = d;
		}
	}
	return idx;
}

TUMDatasetWrapper::TUMDatasetWrapper(std::string dir) : base_dir_(dir), current_id_(0), DataSource()
{
	if (base_dir_.back() != '/')
		base_dir_ += '/';

	LOG(INFO) << "The base directory is : " << base_dir_ << std::endl;
}

bool TUMDatasetWrapper::load_association_file(std::string file_name)
{
	std::ifstream file;
	file.open(base_dir_ + file_name, std::ios_base::in);

	double ts;
	std::string name_depth, name_image;

	while (file >> ts >> name_image >> ts >> name_depth)
	{
		image_name_.emplace_back(name_image);
		depth_name_.emplace_back(name_depth);
		data_ts_.emplace_back(ts);
	}

	file.close();

	if (depth_name_.size() == 0)
	{
		LOG(ERROR) << "Reading images failed, please check your directory.\n";
		return false;
	}
	else
	{
		LOG(INFO) << "Total of " << depth_name_.size() << " Images Loaded.\n";
		return true;
	}
}

void TUMDatasetWrapper::load_ground_truth(std::string file_name)
{
	if (data_ts_.size() == 0)
	{
		LOG(ERROR) << "Please load images first!\n";
		return;
	}

	double ts;
	double tx, ty, tz, qx, qy, qz, qw;

	std::ifstream file;
	file.open(base_dir_ + file_name);

	for (int i = 0; i < 3; ++i)
	{
		std::string line;
		std::getline(file, line);
	}

	std::vector<double> ts_gt;
	std::vector<Sophus::SE3d> vgt;
	while (file >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw)
	{
		Eigen::Quaterniond q(qw, qx, qy, qz);
		q.normalize();
		auto r = q.toRotationMatrix();
		auto t = Eigen::Vector3d(tx, ty, tz);
		Sophus::SE3d gt(r, t);

		ts_gt.push_back(ts);
		vgt.push_back(gt);
	}

	for (int i = 0; i < data_ts_.size(); ++i)
	{
		double time = data_ts_[i];
		int idx = find_closest_index(ts_gt, time);
		groundtruth_ts_.push_back(ts_gt[idx]);
		groundtruth_.push_back(vgt[idx]);
	}

	LOG(INFO) << "Total of " << groundtruth_.size() << " Ground Truth Data Loaded.\n";
	file.close();
}

void TUMDatasetWrapper::save_full_trajectory(std::vector<Sophus::SE3d> full_trajectory) const
{
	std::ofstream file;
	std::string file_path = base_dir_ + "output.txt";
	file.open(file_path, std::ios_base::out);

	for (int i = 0; i < full_trajectory.size(); ++i)
	{
		if (i >= data_ts_.size())
			break;

		double ts = data_ts_[i];
		Sophus::SE3d &curr = full_trajectory[i];
		Eigen::Vector3d t = curr.translation();
		Eigen::Quaterniond q(curr.rotationMatrix());

		file << std::fixed
			 << std::setprecision(4)
			 << ts << " "
			 << t(0) << " "
			 << t(1) << " "
			 << t(2) << " "
			 << q.x() << " "
			 << q.y() << " "
			 << q.z() << " "
			 << q.w() << std::endl;
	}

	file.close();
}

bool TUMDatasetWrapper::read_next_images(cv::Mat &image, cv::Mat &depth)
{
	if (current_id_ >= image_name_.size())
		return false;

	auto image_name = image_name_[current_id_];
	auto depth_name = depth_name_[current_id_];

	image = cv::imread(base_dir_ + image_name, cv::IMREAD_UNCHANGED);
	depth = cv::imread(base_dir_ + depth_name, cv::IMREAD_UNCHANGED);

	current_id_ += 1;
	return true;
}

size_t TUMDatasetWrapper::get_current_id() const
{
	return current_id_;
}

double TUMDatasetWrapper::get_current_timestamp() const
{
	return data_ts_[current_id_];
}

Sophus::SE3d TUMDatasetWrapper::get_current_gt_pose() const
{
	return groundtruth_[current_id_];
}

double TUMDatasetWrapper::get_current_gt_timestamp() const
{
	return groundtruth_ts_[current_id_];
}

std::vector<Sophus::SE3d> TUMDatasetWrapper::get_groundtruth() const
{
	return groundtruth_;
}

float TUMDatasetWrapper::get_depth_scale() const
{
	return 1 / 5000.f;
}

Sophus::SE3d TUMDatasetWrapper::get_initial_pose() const
{
	if (groundtruth_.size() > 0)
		return groundtruth_[0];
	else
		return Sophus::SE3d();
}