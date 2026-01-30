#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <format>
#include <ranges>
#include <print>
#include <numbers>

#include <angles.h>
#include <print_cv_types.h>

using namespace cv;
using namespace std::views;
using namespace std::ranges;
using namespace Eigen;
using std::numbers::pi;
// using std::println;

struct Octave
{
    std::vector<Mat> blurs;
    std::vector<Mat> DoGs;
    std::vector<Mat> vis_DoGs;
    std::vector<double> sigmas;
    std::vector<std::vector<cv::KeyPoint>> keypoints;
};

// intentional copy
std::pair<std::vector<Mat>, std::vector<Mat>> drawKeypointsAndCombine(std::vector<Octave> octaves)
{
    std::vector<Mat> hconcated;
    std::vector<Mat> dogs_hconcated;

    for (int i = 0; i < octaves.size(); i++)
    {
        cv::Mat tmp;
        cv::Mat dog_tmp;
        auto blurs = octaves.at(i).blurs;

        for (auto [idx, img] : enumerate(octaves[i].blurs))
        {
            img.convertTo(img, CV_8U, 255.0);
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
            if (idx == 0 || idx >= octaves[i].DoGs.size() - 1)
            {
                continue;
            }
            if (octaves[i].keypoints[idx].size() > 0)
            {
                std::println("Angle for kypt {}", octaves[i].keypoints[idx][0].angle);
                drawKeypoints(img, octaves[i].keypoints[idx], img, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            }
        }

        hconcat(octaves.at(i).blurs, tmp);
        hconcat(octaves.at(i).vis_DoGs, dog_tmp);
        dogs_hconcated.push_back(dog_tmp);
        hconcated.push_back(tmp);
    }

    return std::make_pair(hconcated, dogs_hconcated);
}

std::pair<Vector3d, Matrix3d> calcGradWHessian(const Mat &img, const Mat &above, const Mat &below,
                                               const double x, const double y)
{
    CV_Assert(img.type() == CV_64F);

    const double Dx = (img.at<double>(y, x + 1) - img.at<double>(y, x - 1)) / 2.0;
    const double Dy = (img.at<double>(y + 1, x) - img.at<double>(y - 1, x)) / 2.0;
    const double Ds = (above.at<double>(y, x) - below.at<double>(y, x)) / 2.0;
    // Dxx=D(x+1)−2D(x)+D(x−1)
    // Dyy=D(y+1)−2D(x)+D(y−1)
    // Dss=D(s+1)−2D(x)+D(s−1)
    // Dxy​=(D(x+1,y+1,s)−D(x−1,y+1,s)−D(x+1,y−1,s)+D(x−1,y−1,s)​) / 4

    const double Dxx = img.at<double>(Point2d(x + 1, y)) - 2 * img.at<double>(y, x) + img.at<double>(Point2d(x - 1, y));
    const double Dyy = img.at<double>(Point2d(x, y + 1)) - 2 * img.at<double>(y, x) + img.at<double>(Point2d(x, y - 1));
    const double Dss = above.at<double>(y, x) - 2 * img.at<double>(y, x) + below.at<double>(y, x);

    const double Dxy = (img.at<double>(Point2d(x + 1, y + 1)) - img.at<double>(Point2d(x - 1, y + 1)) -
                        img.at<double>(Point2d(x + 1, y - 1)) + img.at<double>(Point2d(x - 1, y - 1))) /
                       4.0;

    const double Dxs = (above.at<double>(y, x + 1) - above.at<double>(y, x - 1) - below.at<double>(y, x + 1) + below.at<double>(y, x - 1)) / 4.0;

    const double Dys = (above.at<double>(y + 1, x) - above.at<double>(y - 1, x) - below.at<double>(y + 1, x) + below.at<double>(y - 1, x)) / 4.0;
    // H= ​Dxx​ Dxy​ Dxs​
    //    ​Dxy ​Dyy​ Dys​​
    //    Dxs ​Dys​ Dss​​​
    Eigen::Matrix3d Hessian;
    Hessian << Dxx, Dxy, Dxs,
        Dxy, Dyy, Dys,
        Dxs, Dys, Dss;

    Eigen::Vector3d Gradient;
    Gradient << Dx, Dy, Ds;

    return std::make_pair(Gradient, Hessian);
}

// for negative indexing
template <class Vec>
decltype(auto) py_idx(Vec &v, int i)
{
    i += v.size();
    i %= v.size();
    return v.at(i);
}

std::vector<Octave> run_SIFT(Mat img)
{

    CV_Assert(img.channels() == 3);

    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    img.convertTo(img, CV_64F, 1.0 / 255.0);
    std::cout << "Type: " << type2str(img.type()) << std::endl;
    double scales = 3.0;
    double num_octaves = floor(log2(std::min(img.size().width, img.size().height)));
    double sigma_inital = 1.6;
    double k = pow(2, (1.0 / scales)); // multipler between levels, so that you
    std::cout << "Num octaves " << num_octaves << " Sigma inital " << sigma_inital
              << " K " << k << "\n"
              << "Image size " << img.size()
              << "\n";

    std::vector<Octave> octaves;
    for (int octave_idx = 0; octave_idx < num_octaves - 1; octave_idx++)
    {
        Octave octave;
        Mat base_img;

        if (octave_idx == 0)
        {
            base_img = img.clone();
        }
        else
        {
            // should be at the blur level of 2x the original, then downsample by 2x to get back to 1.6 sigma
            auto prev_blurs = octaves[octave_idx - 1].blurs;
            auto inital = prev_blurs[prev_blurs.size() - 3];
            pyrDown(inital, base_img);
            std::cout << "base Type: " << type2str(inital.type()) << std::endl;
        }

        if (base_img.size().height < 16 || base_img.size().width < 16)
        {
            std::cout << "Image too small, breaking out\n";
            break;
        }

        for (int scale_idx = 0; scale_idx < scales + 3; scale_idx++)
        {
            double sigma = sigma_inital * pow(k, scale_idx);
            Mat blurred_img;
            GaussianBlur(base_img, blurred_img, Size(0, 0), sigma);
            octave.blurs.push_back(blurred_img);
            octave.sigmas.emplace_back(sigma);
        }

        // leave out the first and last element because
        // doing right - left, i.e large blur - small blur
        for (auto [idx, img] : octave.blurs | enumerate | drop(1))
        {
            // std::cout << "idx " << idx << "\n";

            Mat DoG = img - octave.blurs.at(idx - 1);
            std::cout << "Dog Type: " << type2str(DoG.type()) << std::endl;

            octave.DoGs.push_back(DoG);

            // just for debug
            Mat dog_visible;
            double minVal, maxVal;
            cv::minMaxLoc(DoG, &minVal, &maxVal);
            double absMax = std::max(std::abs(minVal), std::abs(maxVal));
            DoG.convertTo(dog_visible, CV_8U, 127.0 / absMax, 128);
            octave.vis_DoGs.push_back(dog_visible);
            octave.keypoints.emplace_back();
            std::cout << "DoG " << idx << " min: " << minVal << " max: " << maxVal << std::endl;
            // std::cout << "Induvidal vis size " << dog_visible.size() << "\n";
        }
        octaves.push_back(octave);
    }

    // Keypoint detection
    // now want to find local maximum/minmums, for each octave, look at all the pixels around it and see
    for (auto &octave : octaves)
    {
        for (auto [idx, img] : octave.DoGs | enumerate)
        {
            // keypoints array already has 5 elements here
            // -1 is correct because DoGs has s + 3 - 1 (5) elements (s=3 normally)
            if (idx == 0 || idx >= octave.DoGs.size() - 1)
            {
                continue;
            }
            const auto &above = octave.DoGs[idx + 1];
            const auto &below = octave.DoGs[idx - 1];

            for (int y = 0; y < img.rows; y++)
            {
                for (int x = 0; x < img.cols; x++)
                {
                    if (y == 0 || y + 1 == img.rows || x == 0 || x + 1 == img.cols)
                    {
                        continue;
                    }

                    bool smallest = true;
                    bool largest = true;
                    double pixel = img.at<double>(y, x);

                    for (const auto dx : views::iota(-1, 2))
                    {
                        for (const auto dy : views::iota(-1, 2))
                        {
                            auto yy = y + dy;
                            auto xx = x + dx;
                            double above_px = above.at<double>(yy, xx);
                            double below_px = below.at<double>(yy, xx);
                            double img_px = img.at<double>(yy, xx);
                            if (std::ranges::max({above_px, below_px, img_px}) > pixel)
                            {
                                largest = false;
                            }

                            if (std::ranges::min({above_px, below_px, img_px}) < pixel)
                            {
                                smallest = false;
                            }
                        }
                    }
                    if (!smallest && !largest)
                    {
                        // std::cout << "passed on pixel\n";
                        continue;
                    }
                    // std::cout << "Found keypoint at " << x << " " << y << "\n";
                    octave.keypoints[idx].emplace_back(KeyPoint(Point2f(x, y), 10.0, -1, pixel));
                    // std::cout << "Keypoint size for idx " << idx << " is " << octave.keypoints[idx].size() << "\n";
                }
            }
        }
    }

    auto [hconcated, dogs_hconcated] = drawKeypointsAndCombine(octaves);
    Mat all_blurs = vstackDiffWidths(hconcated);
    cv::imwrite("all_keypoints_pre_edge_det.png", all_blurs);

    // filtering keypoints, based on strenght of response, as well as if its on a line (i.e curvature along x or y is very diff. than the other)
    /*
    Compute at each keypoint: (thanks claude)
        Dxx = DoG(x+1, y) - 2*DoG(x, y) + DoG(x-1, y)
        Dyy = DoG(x, y+1) - 2*DoG(x, y) + DoG(x, y-1)
        Dxy = (DoG(x+1, y+1) - DoG(x-1, y+1) - DoG(x+1, y-1) + DoG(x-1, y-1)) / 4

        Then:

        Tr = Dxx + Dyy
        Det = Dxx * Dyy - Dxy * Dxy

        Reject if:

        Det <= 0 (saddle point or flat — not a blob)
        Tr² / Det > 12.1 (edge-like, using r=10)

        12.1 comes from
        Tr2​/det=λ1​λ2​(λ1​+λ2​)2​=r(r+1)2​ with r=10
    */
    double threshold_ratio = 10.0;
    double tr_cutoff = pow(threshold_ratio + 1, 2) / threshold_ratio;
    std::cout << "TR cutoff is " << tr_cutoff << " \n";
    for (auto &octave : octaves)
    {
        for (auto [idx, keypoints] : octave.keypoints | enumerate)
        {
            auto inital_len = keypoints.size();
            size_t removed = std::erase_if(keypoints, [&](const cv::KeyPoint &kpt)
                                           {
                                        auto pt = kpt.pt;
                                        auto x = pt.x;
                                        auto y = pt.y;
                                        auto DoG = octave.DoGs[idx]; 
                                        double Dxx = DoG.at<double>(Point2d(x+1, y)) - 2 * DoG.at<double>(pt) + DoG.at<double>(Point2d(x-1, y));
                                        double Dyy = DoG.at<double>(Point2d(x, y+1)) - 2 * DoG.at<double>(pt) + DoG.at<double>(Point2d(x, y-1));
                                        double Dxy = (DoG.at<double>(Point2d(x+1, y+1)) - DoG.at<double>(Point2d(x-1, y+1)) - 
                                                        DoG.at<double>(Point2d(x+1, y-1)) + DoG.at<double>(Point2d(x-1, y-1))) 
                                                        / 4.0;
                                        auto Tr = Dxx + Dyy;
                                        auto Det = Dxx * Dyy - Dxy * Dxy;
                                        if (Det <= 0) return true;
                                        if (((Tr * Tr) / Det) > tr_cutoff) return true;
                                        return false; });
            if (removed)
                std::cout << "Edge detection filtered " << removed << " points out of " << inital_len << " \n";
        }
    }

    std::tie(hconcated, dogs_hconcated) = drawKeypointsAndCombine(octaves);
    all_blurs = vstackDiffWidths(hconcated);
    cv::imwrite("all_keypoints_pre_thresh.png", all_blurs);

    // threshold
    double threshold = 0.03; // from the guys paper, can tune
    for (auto &octave : octaves)
    {
        for (auto &keypoints : octave.keypoints)
        {
            auto inital_len = keypoints.size();
            for (auto &kpt : keypoints)
            {
                // std::cout << "Kpt w Int " << kpt.response << " \n";
            }
            size_t removed = std::erase_if(keypoints, [&](cv::KeyPoint p)
                                           { return std::abs(p.response) < threshold; });
            if (removed)
                std::cout << "Filtered " << removed << " points out of " << inital_len << " \n";
        }
    }

    // now need subpixel refinment, basically imagine point in 3d with the adjacent DoG's above and below
    // you have 26 points around (9 above/below, 8 around), so you have x, y, s (scale, essentially z)
    // you found local a maximum in the keypoint but say in 1d you had
    // imagine 3 pixels in a line 0.5, 0.8, 0.6, we found max at 0.8 but "real" max is actually slightly closer to 0.6 than 0.5 (assuming like a curve, plot on desmos if don't remeber)
    // basically do this for 3d, find gradient, hessian, and solve for what vector gives the optimal offsets
    // if any of those offsets are more than 0.5 pixel away, start from that pixel and try again up to 5 times

    // blurs <0, 1, 2, 3, 4, 5> imgs
    // DoGs    <0, 1, 2, 3, 4>  DoGs
    // Kpts    <0, 1, 2, 3, 4>  Kpts
    // going to end up with 3 levels of usable keypoints
    for (auto &octave : octaves)
    {
        for (auto [idx, keypoints] : octave.keypoints | enumerate)
        {
            // no above/below image to comapare to or no keypoints
            if (idx == 0 || idx == octave.DoGs.size() - 1 || keypoints.empty())
            {
                continue;
            }

            std::vector<cv::KeyPoint> kept;
            for (auto &kpt : keypoints)
            {
                double x = kpt.pt.x;
                double y = kpt.pt.y;
                size_t s_idx = idx; // s will get refined and not become an
                double s = idx;
                bool converged = false;

                for (auto i : views::iota(0, 5))
                {
                    if (s_idx == 0 || s_idx >= octave.DoGs.size() - 1)
                    {
                        std::println("S is out of bounds with val {}", s);
                        break;
                    }
                    const auto &below = octave.DoGs[s_idx - 1];
                    const auto &above = octave.DoGs[s_idx + 1];
                    const auto &img = octave.DoGs[s_idx];

                    auto [Gradient, Hessian] = calcGradWHessian(img, above, below, x, y);
                    // H · Δx = -∇D
                    Eigen::Vector3d result = Hessian.colPivHouseholderQr().solve(-Gradient);
                    if (std::abs(result(0)) > 0.5 || std::abs(result(1)) > 0.5 || std::abs(result(2)) > 0.5)
                    {
                        // move pixels and try again
                        int dx = (result(0) > 0.5) - (result(0) < -0.5);
                        int dy = (result(1) > 0.5) - (result(1) < -0.5);
                        int ds = (result(2) > 0.5) - (result(2) < -0.5);
                        x += dx;
                        y += dy;
                        s_idx += ds;
                        if (ds)
                            std::println("Need to move blur levels {}", ds);
                        std::println("res 1 2 3 {} {} {}", result(0), result(1), result(2));

                        std::println("Dx dy ds {} {} {}", dx, dy, ds);

                        // should maybe go at top of loop since kpt could have started at like 0,0
                        if (x >= img.cols - 1 || y >= img.rows - 1 ||
                            x <= 0 || y <= 0)
                        {
                            std::println("X or Y is too close to edge of the image for refinement");
                            break;
                        }
                    }
                    else
                    {
                        converged = true;
                        std::print("Refined point from (x,y,s)=({}, {}, {})", x, y, s);
                        x += result(0);
                        y += result(1);
                        s = (double)s_idx + result(2);
                        std::println(" to ({}, {}, {})\n", x, y, s);
                        kept.emplace_back(KeyPoint(Point2d(x, y), s, kpt.angle, kpt.response, kpt.octave));
                        break;
                    }
                }
                if (!converged)
                    std::println("Tried 5 times and could not refine keypoint");
            }
            keypoints = std::move(kept);
        }
    }

    // now want to find orientation? of keypoints, angle (0-360) that has which direction the pixels are become the most intense
    // idea is that if you took the same image but rotated 45 degrees, the features should still be able to be matched, the orientation allows you to have say
    // feature A with direction 10deg and then rotated 45 deg still will have 55 deg.
    // importantly when you make the descriptor of the feature, it is relative to this orientation, so the desciptors will both be in the same reference frame regardless of orientation
    for (auto &octave : octaves)
    {
        for (auto [idx, blur_img] : octave.blurs | enumerate)
        {
            // keypoints array already has 5 elements here
            // - 2 comes -1 from size being 1 indexed, -1 from the last element of DoGs doesn't have keypoints since no image to right
            if (idx == 0 || idx >= octave.DoGs.size() - 1)
            {
                continue;
            }

            auto &kpts = octave.keypoints[idx];
            std::vector<cv::KeyPoint> new_keypoints;
            new_keypoints.reserve(kpts.size());

            for (auto &kpt : kpts)
            {
                // does lose refinement (i.e 100.5) but needed to access pixels
                int x = std::round(kpt.pt.x);
                int y = std::round(kpt.pt.y);
                // 1.5 is taken from paper (Lowe 2004)
                // use sigmas stored previously, .size is like essentially just z coord, not how much it was blurred (though related)
                double weight_sigma = 1.5 * octave.sigmas[idx];
                int radius = std::round(weight_sigma * 3.0); // 3 std dev is enough
                std::array<double, 36> orientation_hist{};   // each bin is 10 deg and 36 * 10 = 360

                if (x - radius < 0 || x + radius >= blur_img.cols || y - radius < 0 || y + radius >= blur_img.rows)
                {
                    std::println("Leaving behind kpt due to orientation going out of bounds");
                    continue;
                }

                for (auto dx : views::iota(-radius, radius + 1))
                {
                    for (auto dy : views::iota(-radius, radius + 1))
                    {
                        double grad_x = (blur_img.at<double>(y + dy, x + dx + 1) - blur_img.at<double>(y + dy, x + dx - 1)) / 2.0;
                        double grad_y = (blur_img.at<double>(y + dy + 1, x + dx) - blur_img.at<double>(y + dy - 1, x + dx)) / 2.0;
                        double mag = std::hypot(grad_y, grad_x);
                        double orientation = std::atan2(grad_y, grad_x) + pi; // get it in [0, 2pi]
                        double orientation_deg = angles::to_degrees(orientation);

                        // $$ w_{\text{spatial}}(x, y) = \exp\left(-\frac{(x - x_k)^2 + (y - y_k)^2}{2\sigma_w^2}\right) $$

                        // $$ \sigma_w = 1.5 \times \sigma_{\text{keypoint}} $$
                        int bin = (orientation / (2.0 * pi)) * 36.0;
                        if (bin == 36)
                            bin = 0;
                        bool DEBUG_added_to_bin = false;
                        for (auto bin_idx : {bin - 1, bin, bin + 1})
                        {
                            bin_idx += 36; // don't get trolled by bin_idx = -1 and do -1 % 36
                            bin_idx %= 36;
                            auto mid_angle_deg = (bin_idx + 1) * 10 - 5; // angle between this bin and the next (5, 15, 25, etc)
                            auto mid_angle_rad = angles::from_degrees(mid_angle_deg);
                            auto weight_gaussian = exp(-(pow(dx, 2) + pow(dy, 2)) / (2.0 * pow(weight_sigma, 2))); // closer pixels contribute more
                            auto bin_weighting = (1 - (std::abs(angles::shortest_angular_distance(orientation, mid_angle_rad))) / angles::from_degrees(10.0));
                            // last term is basically farther from actual angle means less weight with a +- of 1 bucket (10 deg)
                            auto to_add = mag * weight_gaussian * bin_weighting;

                            // check more than 0 bcs third term above could be negative
                            if (to_add > 0)
                            {
                                orientation_hist[bin_idx] += to_add;
                                DEBUG_added_to_bin = true;
                            }
                        }
                        assert(DEBUG_added_to_bin);
                    }
                }
                auto arr_enum = orientation_hist | std::views::enumerate;
                // runs the view created by enumerate through max, so that the max element is found with the index
                auto max_it = std::max_element(arr_enum.begin(), arr_enum.end(), [](std::tuple<int, double> a, std::tuple<int, double> b)
                                               { return std::get<1>(a) < std::get<1>(b); });
                auto max_val = std::get<1>(*max_it);
                size_t max_idx = std::get<0>(*max_it);

                std::vector<std::tuple<int, double>> results = {};

                const auto max_percent = 0.8; // 80 % of max also considered valid orientation for keypoint
                const auto &h = orientation_hist;

                // @todo could add check for local max so don't get like three bins in a row that are above >80%
                std::copy_if(arr_enum.begin(), arr_enum.end(), std::back_inserter(results), [&](std::tuple<int, double> a)
                             {const int idx = std::get<0>(a);
                                                const double val = std::get<1>(a);
                                                return ((val >= max_percent * max_val) && 
                                                        (py_idx(h, idx-1) < val) && 
                                                        (py_idx(h, idx+1) < val)); });

                // if (results.size() > 1) std::println("Found at least one 80% keypoint in image");
                // refine bin using left and right ones (i.e can do better than just bin 4, using bins 5 and 3)

                // $$ \Delta b = \frac{h[b-1] - h[b+1]}{2(h[b-1] - 2h[b] + h[b+1])} $$

                // $$ \theta_{\text{refined}} = (b + \Delta b) \cdot 10° = (b + \Delta b) \cdot \frac{\pi}{18} $$

                for (auto result : results)
                {
                    auto b = std::get<0>(result); // idx
                    b += 36;                      // don't get trolled by b = -1 and do -1 % 36
                    b %= 36;
                    const auto dBinNum = h[b - 1] - h[b + 1];
                    const auto dBinDenom = 2.0 * (h[b - 1] - 2.0 * h[b] + h[b + 1]);
                    const auto dBin = dBinNum / dBinDenom;
                    const auto dBin_clamped = std::clamp(dBin, -0.5, 0.5);
                    assert(dBin == dBin_clamped);
                    const auto theta_refined_deg = (b + dBin) * (10.0);
                    std::println("Theta deg {}", theta_refined_deg);
                    assert((theta_refined_deg <= 360.0 && theta_refined_deg >= 0));
                    kpt.angle = theta_refined_deg;
                    kpt.size = blur_img.cols / 20; // just so its actually visible
                    new_keypoints.push_back(kpt);
                }
            }

            std::println("Previous keypoints size {}, new size {} ", kpts.size(), new_keypoints.size());
            kpts = std::move(new_keypoints);
        }
    }

    // now want to actually calculate the feature descriptor, this will be 128d vector that can be used to easily compare against other descriptors
    // idea is simmilar to orientation finder, want to calc gradient magnitude and orientation for each pixel in a radius, 16x16 seems to be used
    // but should be relative to the sigma of the blurred image.

    // then you find the relative angle by subtracting the angle calculated above from the angle just found, i.e orientation - Kpt.angle
    // then rotate each delta pixel to which bins it would have landed in
    // i.e at pixel (4,5) relative to keypoint, usual rotation formula to get x_rotated and y_rotated
    // then figure out which bin that should be assigned to

    // then want to "bin" the results based on both where the pixel is (in the rotated frame), common to use 4x4 x8 bin
    // 4x4 accounts for the pixels (16x16), and  the 8 is bins of 45deg. Don't care for smaller bins because
    // thats the hard part and the rest is pretty easy once all that works

    // 4x4x8 array where 4x4 is for pixel location, 8 is angles 0-360 in 45 degree increments.

    constexpr int NUM_SPATIAL_BINS = 4;
    constexpr int NUM_ANGULAR_BINS = 8; // 45 deg increments

    for (auto &octave : octaves)
    {
        for (auto [idx, blur_img] : octave.blurs | enumerate)
        {
            // keypoints array already has 5 elements here
            // - 2 comes -1 from size being 1 indexed, -1 from the last element of DoGs doesn't have keypoints since no image to right
            if (idx == 0 || idx >= octave.DoGs.size() - 1)
            {
                continue;
            }

            auto &kpts = octave.keypoints[idx];
            for (auto &kpt : kpts)
            {
                std::array<std::array<std::array<double, 8>, NUM_SPATIAL_BINS>, NUM_SPATIAL_BINS> descriptor{};

                int x = std::round(kpt.pt.x);
                int y = std::round(kpt.pt.y);


                const double bin_px_size = 3.0 * octave.sigmas[idx];

                // need the sqrt2 and others to account for worse case rotation (45 deg) and make sure we compute pixels there
                const int radius = bin_px_size * sqrt(2) * ((NUM_SPATIAL_BINS + 1) / 2.0);

                if (x - radius < 0 || x + radius >= blur_img.cols || y - radius < 0 || y + radius >= blur_img.rows)
                {
                    std::println("Leaving behind kpt due to orientation going out of bounds");
                    continue;
                }

                // yes is square, gaussian weighting means so far out does nothing essentially
                for (auto dx : views::iota(-radius, radius + 1))
                {
                    for (auto dy : views::iota(-radius, radius + 1))
                    {
                        // same as above
                        const double grad_x = (blur_img.at<double>(y + dy, x + dx + 1) - blur_img.at<double>(y + dy, x + dx - 1)) / 2.0;
                        const double grad_y = (blur_img.at<double>(y + dy + 1, x + dx) - blur_img.at<double>(y + dy - 1, x + dx)) / 2.0;
                        const double mag = std::hypot(grad_y, grad_x);
                        const double orientation = std::atan2(grad_y, grad_x) + pi; // get it in [0, 2pi]

                        const double theta = angles::from_degrees(kpt.angle);
                        // should keypoint rotation proof, hopefully
                        // only use this for determining which rotation bin it goes in, not rotating to determine spatial bin
                        const double relative_angle = orientation - theta;
                        const double relative_angle_deb = angles::to_degrees(relative_angle);

                        // now need to transform dx dy into rotated coords

                        // $ x_{\text{rot}} = \frac{dx \cos\theta - dy \sin\theta}{w_{\text{bin}}}$  $ y_{\text{rot}} = \frac{dx \sin\theta + dy \cos\theta}{w_{\text{bin}}} $

                        const double dx_rot = (dx * cos(theta) - dy * sin(theta)) / bin_px_size;
                        const double dy_rot = (dx * sin(theta) + dy * cos(theta)) / bin_px_size;

                        // now need to figure out which bin that should go into'
                        // nice claude diagram explaning
                        /*
                      -0.5      0.5      1.5      2.5      3.5
                        │        │        │        │        │
                        ▼        ▼        ▼        ▼        ▼
                        ┌────────┬────────┬────────┬────────┐
                        │        │        │        │        │
                        │ bin 0  │ bin 1  │ bin 2  │ bin 3  │
                        │        │        │        │        │
                        └────────┴────────┴────────┴────────┘
                                          ▲
                                        keypoint
                                        (x_bin = 1.5)
                        */
                        // dx_rot in range (-2 * bin_px_size -> 2 * bin_px_size)
                        // take out bin_px_size and shift right for indexing
                        // likley inspired by opencv impl but seems resonable to take the same path
                        // bins in range [-0.5, 3.5]
                        const double x_bin_cont = (dx_rot / bin_px_size) + 1.5;
                        const double y_bin_cont = (dy_rot / bin_px_size) + 1.5;
                        const double theta_bin_cont = relative_angle * (NUM_ANGULAR_BINS / (2.0 * pi));

                        // sigma = 2 bin widths so 2o^2 = 8
                        const double gaussian_weight = exp(-(pow(dx_rot, 2) + pow(dy_rot, 2)) / 8.0);
                        const double max_contribution = mag * gaussian_weight;
                        double fractional_contribution = 0; // going to check that we do interpolation correctly

                        // trilinear interpolation, a result could end up contributing to 8 cells
                        const int x_bin = floor(x_bin_cont);
                        const int y_bin = floor(y_bin_cont);
                        const int theta_bin = floor(theta_bin_cont);

                        const double x_bin_frac = x_bin_cont - x_bin;
                        const double y_bin_frac = y_bin_cont - y_bin;
                        const double theta_bin_frac = theta_bin_cont - theta_bin;
                        // by flooring and checking +0 and +1 you get both sides that the point could contribute to
                        for (auto dx_bin : {0, 1})
                        {
                            for (auto dy_bin : {0, 1})
                            {
                                for (auto dtheta_bin : {0, 1})
                                {
                                    const int x_test_bin = x_bin + dx_bin;
                                    const int y_test_bin = y_bin + dy_bin;
                                    // just a useful note that this fails if theta_bin + dtheta_bin < -8 
                                    const int theta_test_bin = ((theta_bin + dtheta_bin) + NUM_ANGULAR_BINS) % NUM_ANGULAR_BINS;
                                    if (x_test_bin < 0 || y_test_bin < 0 || x_test_bin > NUM_SPATIAL_BINS-1 || y_test_bin > NUM_SPATIAL_BINS-1) {
                                        continue;
                                    }
                                    const double x_w = (dx_bin == 0) ? 1 - x_bin_frac : x_bin_frac; 
                                    const double y_w = (dy_bin == 0) ? 1 - y_bin_frac : y_bin_frac; 
                                    const double theta_w = (dtheta_bin == 0) ? 1 - theta_bin_frac : theta_bin_frac; 
                                    assert(x_w < 1 && y_w < 1 && theta_w < 1); 
                                    const double to_add = max_contribution * x_w * y_w * theta_w;
                                    assert(to_add >= 0); 
                                    descriptor.at(y_test_bin).at(x_test_bin).at(theta_test_bin) += to_add;
                                    fractional_contribution += to_add;
                                }
                            }
                        }
                        assert(max_contribution + std::numeric_limits<double>::epsilon() >= fractional_contribution);
                    }
                }
            }
        }
    }

    // visualize
    std::tie(hconcated, dogs_hconcated) = drawKeypointsAndCombine(octaves);
    all_blurs = vstackDiffWidths(hconcated);
    cv::imwrite("all_keypoints.png", all_blurs);

    for (int i = 0; i < hconcated.size(); i++)
    {
        imshow(std::format("Size {}", hconcated[i].size().width), hconcated[i]);
    }
    // for (int i = 0; i < dogs_hconcated.size(); i++)
    // {
    //     imshow(std::format("dog Size {}", hconcated[i].size().width), dogs_hconcated[i]);
    // }

    cv::imshow("Original", img);
    waitKey(0);

    return octaves;
}

std::vector<std::vector<Octave>> run_SIFT_batch(std::vector<Mat> images)
{
    std::vector<std::vector<Octave>> result;
    for (auto &img : images)
    {
        result.push_back(run_SIFT(img));
    }
    return result;
}

int main()
{

    std::cout << "Hello" << std::endl;
    std::vector<cv::String> fn;
    glob("images/small*", fn, false);
    // glob("images/gator*", fn, false);

    std::vector<Mat> images;
    size_t count = fn.size();
    for (size_t i = 0; i < count; i++)
    {
        std::cout << "Reading image at " << fn[i] << "\n";
        Mat img = imread(fn[i]);
        images.push_back(img);
        // std::vector<cv::KeyPoint> test{};
        // test.emplace_back(KeyPoint(Point2f(500, 500), img.cols / 20, 90, 0, 2));
        // drawKeypoints(img, test, img, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        // cv::imwrite("angle?.png", img);
        // exit(9);
    }

    std::cout << "Images.size " << images.size() << "\n";
    std::vector<std::vector<Octave>> octaves = run_SIFT_batch(images);

    return 0;
}