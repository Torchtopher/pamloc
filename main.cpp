#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <format>
#include <ranges>

#include <print_cv_types.h>

using namespace cv;
using namespace std::views;
using namespace std::ranges;

struct Octave
{
    std::vector<Mat> blurs;
    std::vector<Mat> DoGs;
    std::vector<Mat> vis_DoGs;
    std::vector<double> sigmas;
    std::vector<std::vector<cv::KeyPoint>> keypoints;
};

// intentional copy 
std::pair<std::vector<Mat>, std::vector<Mat>> drawKeypointsAndCombine(std::vector<Octave> octaves) {
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
            if (octaves[i].keypoints[idx].size() > 0)
            {
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


int main()
{

    std::cout << "Hello" << std::endl;
    std::vector<cv::String> fn;
    //glob("images/small*", fn, false);
    glob("images/feature*", fn, false);

    std::vector<Mat> images;
    size_t count = fn.size();
    for (size_t i = 0; i < count; i++)
    {
        std::cout << "Reading image at " << fn[i] << "\n";
        images.push_back(imread(fn[i]));
    }

    std::cout << "Images.size " << images.size() << "\n";
    std::vector<Octave> octaves;
    // compute blurs for each image
    for (int i = 0; i < images.size(); i++)
    {
        Mat img;
        cv::cvtColor(images[i], img, cv::COLOR_BGR2GRAY);
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
                std::cout << "Blurred Type: " << type2str(blurred_img.type()) << std::endl;

                // std::cout << "Blur " << octave_idx << " Sigma " << sigma << " Img " << blurred_img.size() << std::endl;
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
            // std::cout << "Adding octave at " << octave_idx << "\n";
            // std::cout << "Dog Vis size " << octave.vis_DoGs.size() << "\n";
        }
    }

    // Keypoint detection
    // now want to find local maximum/minmums, for each octave, look at all the pixels around it and see
    for (auto &octave : octaves)
    {
        for (auto [idx, img] : octave.DoGs | enumerate)
        {
            octave.keypoints.emplace_back(); // just keep ones where it has none empty
            if (idx == 0 || idx == octave.DoGs.size() - 1)
            {
                continue;
            }
            const auto& above = octave.DoGs[idx + 1];
            const auto& below = octave.DoGs[idx - 1];

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
                    octave.keypoints[idx].emplace_back(KeyPoint(Point2f(x, y), 5.0, -1, pixel));
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
            for (auto &kpt : keypoints)
            {
                // std::cout << "Kpt w Int " << kpt.response << " \n";
                size_t removed = std::erase_if(keypoints, [&](cv::KeyPoint kpt)
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
                                            return false;
                                        });
                if (removed)
                    std::cout << "Edge detection filtered " << removed << " points out of " << inital_len << " \n";
            }
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

    cv::imshow("Original", images[0]);
    waitKey(0);
    return 0;
}