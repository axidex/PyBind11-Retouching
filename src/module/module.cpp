#include <pybind11/pybind11.h>
#include "ndarray_converter.h"

#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <opencv2/ml.hpp>
#include <wavelib.h>

#include <tinysplinecxx.h>
#include <newfilter.h>

#include <fstream>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include <algorithm>
#include <memory>
#include <functional>
#include <random>

// debugging libs
#include <chrono>
//#include <typeinfo>

template<typename T>
T random(T range_from, T range_to, std::mt19937 generator)
{
    std::uniform_int_distribution<T>    distr(range_from, range_to);
    return distr(generator);
}

cv::Mat GMM(const cv::Mat& final_face, const cv::Mat& src, const cv::Mat& final_face_not)
{
    // Define 5 colors with a maximum classification of no more than 5
    int width = src.cols;
    int height = src.rows;
    int dims = src.channels();

    int nsamples = 2000;
    cv::Mat points = cv::Mat::zeros(nsamples, dims, CV_64FC1);
    cv::Mat labels;
    cv::Mat result = cv::Mat::zeros(src.size(), CV_64FC1);

    // Define classification, that is, how many classification points of function K value
    int num_cluster = 2;
    //printf("num of num_cluster is %d\n", num_cluster);
    // Image RGB pixel data to sample data
    int index = 0;

    int it = 0;
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    while (points.at<double>(999, 0) == 0 && points.at<double>(999, 1) == 0 && points.at<double>(999, 2) == 0)
    {
        int rand_width = random<int>(0, width-1, generator);
        int rand_height = random<int>(0, height-1, generator);

        int blue = final_face.at<cv::Vec3b>(rand_height, rand_width)[0];
        int green = final_face.at<cv::Vec3b>(rand_height, rand_width)[1];
        int red = final_face.at<cv::Vec3b>(rand_height, rand_width)[2];

        if (blue != 0 || green != 0 || red != 0)
        {
            points.at<double>(it, 0) = static_cast<double>(blue);
            points.at<double>(it, 1) = static_cast<double>(green);
            points.at<double>(it, 2) = static_cast<double>(red);
            it++;
        }
    }

    while (points.at<double>(1999, 0) == 0 && points.at<double>(1999, 1) == 0 && points.at<double>(1999, 2) == 0)
    {
        int rand_width = random<int>(0, width - 1, generator);
        int rand_height = random<int>(0, height - 1, generator);

        int blue = final_face_not.at<cv::Vec3b>(rand_height, rand_width)[0];
        int green = final_face_not.at<cv::Vec3b>(rand_height, rand_width)[1];
        int red = final_face_not.at<cv::Vec3b>(rand_height, rand_width)[2];

        if (blue != 0 || green != 0 || red != 0)
        {
            points.at<double>(it, 0) = static_cast<double>(blue);
            points.at<double>(it, 1) = static_cast<double>(green);
            points.at<double>(it, 2) = static_cast<double>(red);
            it++;
        }
    }
    // EM Cluster Train
    cv::Ptr<cv::ml::EM> em_model = cv::ml::EM::create();
    // Partition number
    em_model->setClustersNumber(num_cluster);
    // Set covariance matrix type
    em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_SPHERICAL);
    // Set convergence conditions
    em_model->setTermCriteria(
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.1));
    // Store the probability partition to labs EM according to the sample training
    em_model->trainEM(points, cv::noArray(), labels, cv::noArray());
    // Mark color and display for each pixel
    cv::Mat sample(1, dims, CV_64FC1); //

    std::vector<cv::Mat> cov;

    cv::Mat weights = em_model->getWeights();
    cv::Mat means = em_model->getMeans();
    em_model->getCovs(cov);
    std::cout << "mean:" << means.size << " weights:" << weights.size << " cov:" << cov.size() << " x " << cov[0].size << std::endl;
    int r = 0, g = 0, b = 0;
    std::vector<double> determinant_cov_mat_sqrt;
    std::vector<cv::Mat> inv_cov_mat;
    
    for (int i = 0; i < cov.size(); ++i)
    {
        cv::Mat cov_mat = cov[i];
        inv_cov_mat.emplace_back(cov_mat.inv());
        determinant_cov_mat_sqrt.emplace_back(std::pow(cv::determinant(cov_mat), 0.5));
    }
    double pi_dims = pow(2 * 3.1415926, dims/2);
    // Put each pixel in the sample
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            // Get the color of each channel
            b = final_face.at<cv::Vec3b>(row, col)[0];
            g = final_face.at<cv::Vec3b>(row, col)[1];
            r = final_face.at<cv::Vec3b>(row, col)[2];
            if (b == 0 && g == 0 && r == 0)
                continue;
            // Put pixels in sample data
            sample.at<double>(0, 0) = static_cast<double>(b);
            sample.at<double>(0, 1) = static_cast<double>(g);
            sample.at<double>(0, 2) = static_cast<double>(r);

            double prob = 0;

            for (int cluster = 0; cluster < cov.size(); ++cluster)
            {
                //cv::Mat means_vec(1, dims, CV_64FC1);
                //cv::Mat cov_mat(dims, dims, CV_64FC1);
                cv::Mat means_vec = means.row(cluster);
                double tmp_first = 1 / (pi_dims * determinant_cov_mat_sqrt[cluster]);
                cv::Mat tmp_second = -0.5 * ((sample - means_vec) * inv_cov_mat[cluster] * (sample - means_vec).t());
                cv::exp(tmp_second, tmp_second);

                if (cluster == 0)
                    prob += weights.at<double>(0, cluster) * tmp_first * tmp_second.at<double>(0, 0);
                else
                    prob -= weights.at<double>(0, cluster) * tmp_first * tmp_second.at<double>(0, 0);
                if (prob < 0)
                    prob = 0;
            }

            //double response = em_model->predict2(sample, cv::noArray())[0];
            //double prob = exp(response);

            result.at<double>(row, col) = prob;
        }
    }

    return result.clone();
}

tinyspline::BSpline GetCurve(int n, int nend, const dlib::full_object_detection& shape)
{
    std::vector<tinyspline::real> knots;
    knots.clear();

    for (int i = n; i < nend + 1; ++i) {
        const auto& point = shape.part(i);
        knots.push_back(point.x());
        knots.push_back(point.y());
    }

    // Make a closed curve
    knots.emplace_back(knots[0]);
    knots.emplace_back(knots[1]);
    // Interpolate the curve
    auto spline = tinyspline::BSpline::interpolateCubicNatural(knots, 2);

    return spline;
}

void DrawLandmark(double x, double y, cv::Mat& landmark_image)
{
    constexpr auto radius = 5;
    const auto color = cv::Scalar(0, 255, 255);
    constexpr auto thickness = 2;
    const auto center = cv::Point(x, y);
    cv::circle(landmark_image, center, radius, color, thickness);
}

void FacePart(cv::Mat& mask_image, const dlib::shape_predictor landmark_detector, const dlib::full_object_detection shape, int entry, int end, cv::Mat& landmark_image)
{
    std::vector<tinyspline::real> knots;

    // Right eye cubic curve
    const auto Curve = GetCurve(entry, end, shape);
    // Sample landmark points from the curve
    std::array<cv::Point, size_t(25)> Pts;
    for (int i = 0; i < 25; ++i) {
        const auto net = Curve(1.0 / 25 * i);
        const auto result = net.result();
        const auto x = result[0], y = result[1];
        DrawLandmark(x, y, landmark_image);
        Pts[i] = cv::Point(x, y);
    }
    // Draw binary mask
    cv::fillConvexPoly(mask_image, Pts, cv::Scalar(255), cv::LINE_AA);
}

bool areEqual(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat temp;
    cv::bitwise_xor(a, b, temp);
    return !(cv::countNonZero(temp.reshape(1)));
}

std::vector <std::vector<cv::Mat>> MaskGenerate(const cv::Mat& src, const std::string& model_dir)
{
    std::vector<cv::Mat> mini_faces;
    const auto input_img = src;
    // Make a copy for drawing landmarks
    cv::Mat landmark_image = input_img.clone();
    // Make a copy for drawing binary mask
    cv::Mat mask_image = cv::Mat::zeros(input_img.size(), CV_8UC1);

    auto landmark_model_path = cv::samples::findFile(model_dir, /*required=*/false);
    if (landmark_model_path.empty()) {
        std::cout << "Could not find the landmark model file: " << model_dir << "\n"
            << "The model should be located in `models_dir`.\n";
        throw std::invalid_argument("Landmark model path is empty");
    }

    // Leave the original input image untouched
    cv::Mat work_image = input_img.clone();

    dlib::shape_predictor landmark_detector;
    dlib::deserialize(landmark_model_path) >> landmark_detector;

    // Detect faces
    // Need to use `dlib::cv_image` to bridge OpenCV and dlib.
    const auto dlibImg = dlib::cv_image<dlib::bgr_pixel>(input_img);
    auto faceDetector = dlib::get_frontal_face_detector();
    auto faces = faceDetector(dlibImg);

    // Draw landmark on the input image

    // clang-format off
    // Get outer contour of facial features
    // The 68 facial landmark from the iBUG 300-W dataset(https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
    // Jaw:              0-16 (lower face boundary)
    // Right eyebrow:   17-21
    // Left eyebrow:    22-26 
    // Nose:            27-35
    // Right eye:       36-41
    // Left eye:        42-47
    // Mouth:           48-67 (boundary:48-59)
    // clang-format on
    std::vector<cv::Mat> face_masks;
    for (const auto& face : faces) {
        std::vector<tinyspline::real> knots;
        const auto shape = landmark_detector(dlibImg, face);

        FacePart(mask_image, landmark_detector, shape, 36, 41, landmark_image); // right eye
        FacePart(mask_image, landmark_detector, shape, 42, 47, landmark_image); // left eye
        //FacePart(mask_image, landmark_detector, shape, 48, 59, landmark_image);

        const auto mouth_curve = GetCurve(48, 59, shape);
        constexpr auto mouth_point_number = 40;
        std::array<cv::Point, mouth_point_number> mouth_points;
        // Sample landmark points from the curve
        for (int i = 0; i < mouth_point_number; ++i) {
            const auto net = mouth_curve(1.0 / mouth_point_number * i);
            const auto result = net.result();
            const auto x = result[0], y = result[1];
            DrawLandmark(x, y, landmark_image);
            mouth_points[i] = cv::Point(x, y);
        }
        // Draw binary mask
        cv::fillPoly(mask_image, mouth_points, cv::Scalar(255), cv::LINE_AA);
        // Estimate an ellipse that can complete the upper face region
        constexpr auto nJaw = 17;
        std::vector<cv::Point> lower_face_points;
        for (int i = 0; i < nJaw; ++i) {
            const auto& point = shape.part(i);
            const auto x = point.x(), y = point.y();
            DrawLandmark(x, y, landmark_image);
            lower_face_points.push_back(cv::Point(x, y));
        }
        // Guess a point located in the upper face region
        // Pb: 8 (bottom of jaw)
        // Pt: 27 (top of nose)
        const auto& Pb = shape.part(8);
        const auto& Pt = shape.part(27);
        const auto x = Pb.x();
        const auto x_nose = Pt.x();
        std::cout << abs(x - x_nose) << std::endl;
        const auto y = Pt.y() - 0.85 * abs(Pb.y() - Pt.y());
        DrawLandmark(x, y, landmark_image);
        lower_face_points.push_back(cv::Point(x, y));
        /* TODO solution for upper dot
        cv::Point center(x, y);//Declaring the center point
        int radius = 25; //Declaring the radius
        cv::Scalar line_Color(173, 255, 47);//Color of the circle
        int thickness = 2;//thickens of the line
        cv::circle(work_image, center, radius, line_Color, thickness);
        cv::imshow("mat", work_image);//Showing the circle//
        cv::waitKey(0);//Waiting for Keystroke//
        */
        // Fit ellipse
        const auto box = cv::fitEllipseDirect(lower_face_points);
        cv::Mat mask_tmp = cv::Mat(mask_image.size(), CV_8UC1, cv::Scalar(255));
        cv::ellipse(mask_tmp, box, cv::Scalar(0), /*thickness=*/-1, cv::FILLED);

        cv::bitwise_or(mask_tmp, mask_image, mask_image);
        cv::bitwise_not(mask_image, mask_image);
     
        face_masks.push_back(mask_image.clone());

        // Creating cropped mini face for GMM
        cv::Mat mini_face = input_img(cv::Range(face.top(), face.bottom()), cv::Range(face.left(), face.right()));
        cv::Mat mini_mask = mask_image(cv::Range(face.top(), face.bottom()), cv::Range(face.left(), face.right()));
        cv::Mat mini_final;

        cv::Mat mini_mask_channels[3] = { mini_mask, mini_mask, mini_mask };
        cv::Mat mini_mask_img_3_channels;
        cv::merge(mini_mask_channels, 3, mini_mask_img_3_channels);

        cv::bitwise_and(mini_face, mini_mask_img_3_channels, mini_final);
        mini_faces.emplace_back(mini_final);
    }

    // TODO 
    /*
    cv::Mat zero_like = cv::Mat::zeros(face_masks[0].rows, face_masks[0].cols, CV_8UC1);
    for (int face = 0; face < face_masks.size(); ++face)
    {
        for (int face2 = 0; face2 < face_masks.size(); ++face2)
        {
            if (face == face2)
                continue;
            cv::Mat face_and;
            cv::bitwise_and(face_masks[face], face_masks[face2], face_and);
            bool isEqual = areEqual(zero_like, face_and);
            //bool isEqual = !cv::norm(zero_like, face_and, cv::NORM_L1);
            if (!isEqual)
                cv::bitwise_xor(face_masks[face], face_masks[face2], face_masks[face]);
        }
        cv::imshow("tmp", face_masks[face]);
        cv::waitKey();
    }
    */
    std::vector<cv::Mat> final_face_vec, final_face_not_vec, mask_img_not_vec, input_img_vec, probability_mask_vec, final_mask_vec, final_mask_not_vec;
    input_img_vec.emplace_back(input_img.clone());
    for (int face = 0; face < face_masks.size(); ++face)
    {
        cv::Mat mask_channels[3] = { face_masks[face], face_masks[face], face_masks[face] };
        cv::Mat mask_img_3_channels;
        cv::merge(mask_channels, 3, mask_img_3_channels);
        cv::Mat spot_image_temp;
        cv::Mat mask_img_not, mask_GF;
        // Guided Filter
        cv::bitwise_and(input_img, mask_img_3_channels, spot_image_temp);
        cv::ximgproc::guidedFilter(spot_image_temp, mask_img_3_channels, mask_GF, 10, 200); //10 200
        cv::bitwise_not(mask_GF, mask_img_not);
        mask_img_not_vec.push_back(mask_img_not.clone());

        // Inner mask
        cv::Mat mask_morphology_Ex;
        cv::Mat maskElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(30, 30)); //71 71

        cv::morphologyEx(face_masks[face], mask_morphology_Ex, cv::MORPH_ERODE, maskElement);
        cv::Mat mask_morphology_Exs[3] = { mask_morphology_Ex, mask_morphology_Ex, mask_morphology_Ex };
        cv::Mat mask_morphology_Ex_3_channels;
        cv::merge(mask_morphology_Exs, 3, mask_morphology_Ex_3_channels);

        // Make a preserved image for future use
        cv::Mat preserved_image, mask_preserved;
        cv::bitwise_not(mask_morphology_Ex_3_channels, mask_preserved);
        cv::bitwise_and(work_image, mask_preserved, preserved_image);

        // Spot Concealment
        // Convert the RGB image to a single channel gray image
        cv::Mat gray_image;
        cv::cvtColor(work_image, gray_image, cv::COLOR_BGR2GRAY);

        // Compute the DoG to detect edges
        cv::Mat blur_image_1, blur_image_2, DoG_image;
        const auto sigmaY = gray_image.cols / 200.0;
        const auto sigmaX = gray_image.rows / 200.0;
        cv::GaussianBlur(gray_image, blur_image_1, cv::Size(3, 3), /*sigma=*/0);
        cv::GaussianBlur(gray_image, blur_image_2, cv::Size(0, 0), sigmaX, sigmaY);
        cv::subtract(blur_image_2, blur_image_1, DoG_image);
        cv::Mat not_DoG_image;
        cv::bitwise_not(DoG_image, not_DoG_image);

        // Apply binary mask to the image
        cv::Mat not_DoG_images[3] = { not_DoG_image, not_DoG_image, not_DoG_image };
        cv::Mat not_DoG_image_3_channels;
        cv::merge(not_DoG_images, 3, not_DoG_image_3_channels);

        // Creating final mask
        cv::Mat final_mask, final_mask_not;

        cv::bitwise_and(mask_GF, not_DoG_image_3_channels, final_mask);

        probability_mask_vec.emplace_back(final_mask.clone());
        cv::threshold(final_mask, final_mask, 230, 255, cv::THRESH_BINARY);

        cv::bitwise_not(final_mask, final_mask_not);
        // Creating final face and not face
        cv::Mat final_face_not, final_face;
        cv::bitwise_and(work_image, final_mask, final_face);
        final_face_vec.emplace_back(final_face.clone());
        final_mask_vec.emplace_back(final_mask.clone());
        cv::bitwise_and(work_image, final_mask_not, final_face_not);
        final_face_not_vec.emplace_back(final_face_not.clone());
        final_mask_not_vec.emplace_back(final_mask_not.clone());
    }

    return std::vector({ final_face_vec, final_face_not_vec, mask_img_not_vec, input_img_vec, probability_mask_vec, final_mask_vec, final_mask_not_vec, mini_faces });
}

std::vector<std::vector<cv::Mat>> MaskGenerate(const std::string& image_dir, const std::string& model_dir)
{
    const auto input_img =
        cv::imread(cv::samples::findFile(image_dir, /*required=*/false, /*silentMode=*/true));
    if (input_img.empty()) {
        std::cout << "Could not open or find the image: " << image_dir << "\n"
            << "The image should be located in `images_dir`.\n";
        throw std::invalid_argument("Input image path is empty");
    }
    return MaskGenerate(input_img, model_dir);
}

std::vector<cv::Mat> Smoothing(std::vector<cv::Mat>& final_face, const std::vector<cv::Mat>& final_face_not, const std::vector<cv::Mat>& mask_img_not, const cv::Mat& input_img)
{
    std::vector<cv::Mat> smoothed_faces;
    for (int face = 0; face < final_face.size(); ++face)
    {
        cv::Mat tmp1, tmp2;
        cv::Mat no_face;
        int dx = 5; // 5
        double fc = 50; // 50 15
        //JointWMF filter;
        //tmp1 = filter.filter(final_face[face], final_face[face], dx, fc);
        bilateralFilter(final_face[face], tmp1, dx, fc, fc);

        cv::bitwise_and(input_img, mask_img_not[face], no_face);

        cv::add(final_face_not[face], tmp1, tmp2);
        //dst = filter.filter(tmp2, tmp2, 2, 10);
        //bilateralFilter(tmp2, dst, 5, 20, 20);
        smoothed_faces.emplace_back(tmp2);
    }

    return smoothed_faces;
}

std::vector<cv::Mat> Smoothing(std::vector<std::vector<cv::Mat>>& masks)
{
    return Smoothing(masks[0], masks[1], masks[2], masks[3][0]);
}

struct Deleter {
    void operator() (wave_set* f)
    {
        wave_free(f);
    }

    void operator() (wt2_set* f)
    {
        wt2_free(f);
    }

    void operator() (double* f)
    {
        free(f);
    }
};

std::vector < std::vector<double >> CalculateCoef(const std::vector<cv::Mat>& probability_masks, const cv::Mat& original, const std::vector<cv::Mat>& smoothed)
{
    std::vector<std::vector<double>> faces_coefs;
    for (int face = 0; face < probability_masks.size(); ++face)
    {
        std::vector<double> coefs; // bgr
        double gamma = 0.5;
        double beta = 1;
        cv::Mat probability_mask, smoothed_face;

        cv::Mat bgr_channels_smoothed[3], bgr_channels_orig[3];
        cv::Mat bgr_channels_mask[3];
        cv::split(original, bgr_channels_orig);

        smoothed_face = smoothed[face];
        cv::split(smoothed_face.clone(), bgr_channels_smoothed);
        probability_mask = probability_masks[face];

        for (int color = 0; color < 3; ++color)
        {
            cv::Mat original_color = bgr_channels_orig[color];
            cv::Mat smoothed_color = bgr_channels_smoothed[color];
            cv::Mat divided_original_color = original_color / 255;
            cv::Mat divided_smoothed_color = smoothed_color / 255;

            cv::Mat subtracted_abs_original_smoothed = cv::abs(original_color - smoothed_color);
            cv::Mat double_subtracted_abs_original_smoothed;
            subtracted_abs_original_smoothed.convertTo(double_subtracted_abs_original_smoothed, CV_64FC1);
            cv::Mat product_probs_intensity = double_subtracted_abs_original_smoothed.mul(probability_mask);

            double coef = cv::sum(product_probs_intensity)[0];

            coef = ((gamma * coef) / (255)) + beta;
            std::cout << coef << std::endl;
            coefs.emplace_back(coef);
        }
        faces_coefs.emplace_back(coefs);
    }


    return faces_coefs;
}

std::vector<std::vector<cv::Mat>> Restore(const cv::Mat& orig, const std::vector<cv::Mat>& smootheds, std::vector<std::vector<double>> coefs)
{
    std::vector<std::vector<cv::Mat>> restored_faces;
    for (int face = 0; face < smootheds.size(); ++face)
    {
        cv::Mat smoothed = smootheds[face];
        cv::Mat bgr_channels_smoothed[3], bgr_channels_orig[3];
        //if (alfa > 1 || alfa < 0)
        //    throw std::invalid_argument("Alfa should be between 0 and 1");
        cv::split(smoothed.clone(), bgr_channels_smoothed);
        cv::split(orig.clone(), bgr_channels_orig);
        const int kLevel = 3;
        cv::Mat double_smoothed, double_orig;
        int N = smoothed.rows * smoothed.cols;
        std::vector<cv::Mat> colors;
        for (int color = 0; color < 3; ++color)
        {
            double alfa = coefs[face][color];
            double beta = 1 - alfa;
            bgr_channels_smoothed[color].convertTo(double_smoothed, CV_64F);
            bgr_channels_orig[color].convertTo(double_orig, CV_64F);
            double* color_smoothed = double_smoothed.ptr<double>(0);
            double* color_orig = double_orig.ptr<double>(0);

            const char* name = "db2";
            std::unique_ptr <wave_set, Deleter> obj_smoothed(wave_init(name), Deleter());
            std::unique_ptr <wave_set, Deleter> obj_orig(wave_init(name), Deleter());

            std::unique_ptr <wt2_set, Deleter> wt_smoothed(wt2_init(obj_smoothed.get(), "dwt", smoothed.rows, smoothed.cols, kLevel), Deleter());
            std::unique_ptr <wt2_set, Deleter> wt_original(wt2_init(obj_orig.get(), "dwt", orig.rows, orig.cols, kLevel), Deleter());

            constexpr size_t kTotalParts = 3;
            struct {
                int row;
                int columns;
            } parts_smoothed[kTotalParts], parts_original[kTotalParts];

            std::unique_ptr <double, Deleter> wavecoeffs_orig(dwt2(wt_original.get(), color_orig), Deleter());
            std::unique_ptr <double, Deleter> wavecoeffs_smoothed(dwt2(wt_smoothed.get(), color_smoothed), Deleter());

            std::vector<cv::Mat> coefficients_HHi_original_mat, coefficients_HHi_smoothed_mat;
            for (int i = 0; i < kLevel; ++i)
            {
                coefficients_HHi_original_mat.emplace_back(cv::Mat(parts_original[i].row, parts_original[i].columns, CV_64F, getWT2Coeffs(wt_original.get(), wavecoeffs_orig.get(), i + 1, 'D', &parts_original[i].row, &parts_original[i].columns)));
                coefficients_HHi_smoothed_mat.emplace_back(cv::Mat(parts_original[i].row, parts_original[i].columns, CV_64F, getWT2Coeffs(wt_smoothed.get(), wavecoeffs_smoothed.get(), i + 1, 'D', &parts_original[i].row, &parts_original[i].columns)));
                cv::addWeighted(coefficients_HHi_original_mat[i], alfa, coefficients_HHi_smoothed_mat[i], beta, 0, coefficients_HHi_smoothed_mat[i]);
                if (i == kLevel - 1)
                    break;
                alfa = alfa * 0.5;
                beta = 1 - alfa;
            }

            cv::Mat output_matrix = cv::Mat::zeros(smoothed.rows, smoothed.cols, CV_64F);

            double* output = output_matrix.ptr<double>(0);

            idwt2(wt_smoothed.get(), wavecoeffs_smoothed.get(), output);

            colors.push_back(output_matrix);
        }
        cv::Mat converted_matrix_blue, converted_matrix_green, converted_matrix_red;
        colors[0].convertTo(converted_matrix_blue, CV_8U);
        colors[1].convertTo(converted_matrix_green, CV_8U);
        colors[2].convertTo(converted_matrix_red, CV_8U);
        restored_faces.emplace_back(std::vector({ converted_matrix_blue, converted_matrix_green, converted_matrix_red }));
    }

    return restored_faces;
}

std::vector<cv::Mat> ProbMaskGenerate(const std::vector<cv::Mat>& final_face, const cv::Mat& orig, const std::vector<cv::Mat>& final_face_not)
{

    std::vector<cv::Mat> gmm_masks;

    for (int face = 0; face < final_face.size(); ++face)
    {
        gmm_masks.emplace_back(GMM(final_face[face], orig, final_face_not[face]));
    }

    /*for (int face = 0; face < probability_masks.size(); ++face)
    {
        cv::Mat bgr_channels_mask[3];
        cv::Mat gmm_copy = gmm.clone();
        cv::split(probability_masks[face].clone(), bgr_channels_mask);
        int width = gmm.rows;
        int height = gmm.cols;
        cv::Mat tmp_mask;
        tmp_mask = bgr_channels_mask[0];

        parallel_for_(cv::Range(0, width * height), [&](const cv::Range& range) {
            for (int r = range.start; r < range.end; r++)
            {
                if (tmp_mask.at<uchar>(r) == 0)
                {
                    gmm_copy.at<double>(r) = 0;
                }
            }
            });
        gmm_masks.emplace_back(gmm_copy.clone());
    }*/
    return gmm_masks;
}

cv::Mat Retouching(const cv::Mat& src, const std::string& model_dir) {
    //std::chrono::duration<double> elapsed_seconds = end - start;
    //std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    auto start_MaskGenerate = std::chrono::steady_clock::now();
    std::vector < std::vector<cv::Mat> > masks_by_faces = MaskGenerate(src, model_dir);
    auto end_MaskGenerate = std::chrono::steady_clock::now();
    for (auto face : masks_by_faces[7])
    {
        cv::imshow("mat", face);
        cv::waitKey();
        cv::destroyAllWindows();
    }

    std::vector <cv::Mat> probability_masks = masks_by_faces[4];
    cv::Mat orig = masks_by_faces[3][0];
    auto start_ProbMaskGenerate = std::chrono::steady_clock::now();
    std::vector<cv::Mat> gmm_masks = ProbMaskGenerate(masks_by_faces[0], orig, masks_by_faces[1]);
    auto end_ProbMaskGenerate = std::chrono::steady_clock::now();

    auto start_Smoothing = std::chrono::steady_clock::now();
    std::vector <cv::Mat> smoothed = Smoothing(masks_by_faces);
    auto end_Smoothing = std::chrono::steady_clock::now();
    /*for (int i = 0; i < smoothed.size(); ++i)
    {
        cv::imshow("smoothed" + std::to_string(i) , masks_by_faces[0][i]);
    }
    cv::waitKey();
    cv::destroyAllWindows();
    */
    auto start_CalculateCoef = std::chrono::steady_clock::now();
    std::vector<std::vector<double>> coefs = CalculateCoef(gmm_masks, orig, smoothed);
    auto end_CalculateCoef = std::chrono::steady_clock::now();

    auto start_Restore = std::chrono::steady_clock::now();
    std::vector<std::vector<cv::Mat>> faces(Restore(orig, smoothed, coefs));
    auto end_Restore = std::chrono::steady_clock::now();
    std::vector<cv::Mat> restored_faces, restored_faces_by_mask;
    std::chrono::duration<double> elapsed_seconds = end_MaskGenerate - start_MaskGenerate;
    std::cout << "elapsed time MaskGenerate: " << elapsed_seconds.count() << "s\n";
    elapsed_seconds = end_ProbMaskGenerate - start_ProbMaskGenerate;
    std::cout << "elapsed time ProbMaskGenerate: " << elapsed_seconds.count() << "s\n";
    elapsed_seconds = end_Smoothing - start_Smoothing;
    std::cout << "elapsed time Smoothing: " << elapsed_seconds.count() << "s\n";
    elapsed_seconds = end_CalculateCoef - start_CalculateCoef;
    std::cout << "elapsed time CalculateCoef: " << elapsed_seconds.count() << "s\n";
    elapsed_seconds = end_Restore - start_Restore;
    std::cout << "elapsed time Restore: " << elapsed_seconds.count() << "s\n";
    auto start_PostProcessing = std::chrono::steady_clock::now();
    for (auto face : faces)
    {
        cv::Mat final_clrs;
        cv::merge(cv::_InputArray(face), final_clrs);
        restored_faces.emplace_back(final_clrs);
    }

    for (int face = 0; face < restored_faces.size(); ++face)
    {
        cv::Mat restored_face;
        cv::bitwise_and(restored_faces[face], masks_by_faces[5][face], restored_face);
        restored_faces_by_mask.emplace_back(restored_face);
    }
    cv::Mat not_face = orig.clone();
    for (auto mask : masks_by_faces[6])
    {
        cv::bitwise_and(not_face, mask, not_face);
    }

    for (auto face : restored_faces_by_mask)
    {
        cv::add(not_face, face, not_face);
    }
    auto end_PostProcessing = std::chrono::steady_clock::now();
    elapsed_seconds = end_PostProcessing - start_PostProcessing;
    std::cout << "elapsed time PostProcessing: " << elapsed_seconds.count() << "s\n";
    return not_face;
}

cv::Mat RetouchingImg(const std::string& image_dir, const std::string& model_dir) {
    const auto input_img =
        cv::imread(cv::samples::findFile(image_dir, /*required=*/false, /*silentMode=*/true));
    if (input_img.empty()) {
        std::cout << "Could not open or find the image: " << image_dir << "\n"
            << "The image should be located in `images_dir`.\n";
        throw std::invalid_argument("Input image path is empty");
    }
    return Retouching(input_img, model_dir);
}

void Show(cv::Mat mat)
{
    cv::imshow("mat", mat);
    cv::waitKey();
    cv::destroyAllWindows();
}

namespace py = pybind11;

PYBIND11_MODULE(smoothingmodule, module)
{
    NDArrayConverter::init_numpy();

    module.doc() = "smoothingmodule";

    module.def("Retouching", &Retouching, "Retouching function\n\nArguments:\n\tsrc : array like\n\tinput image\nmodelDir : string\n\t path to the your model\n\nReturns\noutput :array like\n\toutput image"
        , py::arg("src"), py::arg("model_dir"));
    module.def("RetouchingImg", &RetouchingImg, "Retouching function\n\nArguments:\n\timgDir : string\n\tpath to input image\nmodelDir : string\n\t path to the your model\n\nReturns\noutput : array like\n\toutput image"
        , py::arg("image_dir"), py::arg("model_dir"));
    module.def("Show", &Show, "imshow", py::arg("mat"));
}