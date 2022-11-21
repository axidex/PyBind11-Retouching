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

// debugging libs
//#include <chrono>
//#include <typeinfo>

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

std::vector <std::vector<cv::Mat>> MaskGenerate(const cv::Mat& src, const std::string& model_dir)
{
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
        // Pt: 27 (top of nose
        const auto& Pb = shape.part(8);
        const auto& Pt = shape.part(27);
        const auto x = Pb.x();
        const auto y = Pt.y() - 0.85 * abs(Pb.y() - Pt.y());
        DrawLandmark(x, y, landmark_image);
        lower_face_points.push_back(cv::Point(x, y));
        // Fit ellipse
        const auto box = cv::fitEllipseDirect(lower_face_points);
        cv::Mat mask_tmp = cv::Mat(mask_image.size(), CV_8UC1, cv::Scalar(255));
        cv::ellipse(mask_tmp, box, cv::Scalar(0), /*thickness=*/-1, cv::FILLED);
        

        cv::bitwise_or(mask_tmp, mask_image, mask_image);
        cv::bitwise_not(mask_image, mask_image);
        face_masks.push_back(mask_image.clone());
    }

    std::vector<cv::Mat> final_face_vec, final_face_not_vec, mask_img_not_vec, input_img_vec, probability_mask_vec, final_mask_vec, final_mask_not_vec;
    input_img_vec.emplace_back(input_img.clone());
    for (int face = 0; face < face_masks.size(); ++face)
    {
        cv::Mat mask_channels[3] = { face_masks[face], face_masks[face], face_masks[face] };
        cv::Mat mask_img_3_channels;
        cv::merge(mask_channels, 3, mask_img_3_channels);
        cv::Mat spot_image, spot_image_temp;
        cv::Mat mask_img_not, mask_GF;

        cv::bitwise_and(input_img, mask_img_3_channels, spot_image_temp);

        cv::ximgproc::guidedFilter(spot_image_temp, mask_img_3_channels, mask_GF, 10, 200); //10 200
        cv::bitwise_not(mask_GF, mask_img_not);
        mask_img_not_vec.push_back(mask_img_not.clone());

        cv::bitwise_and(input_img, mask_GF, spot_image);

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

        cv::Mat final_mask, final_mask_not;

        cv::bitwise_and(mask_GF, not_DoG_image_3_channels, final_mask);

        probability_mask_vec.emplace_back(final_mask.clone());
        cv::threshold(final_mask, final_mask, 230, 255, cv::THRESH_BINARY);

        cv::bitwise_not(final_mask, final_mask_not);
        cv::Mat final_face_not, final_face;
        cv::bitwise_and(work_image, final_mask, final_face);
        final_face_vec.emplace_back(final_face.clone());
        final_mask_vec.emplace_back(final_mask.clone());
        cv::bitwise_and(work_image, final_mask_not, final_face_not);
        final_face_not_vec.emplace_back(final_face_not.clone());
        final_mask_not_vec.emplace_back(final_mask_not.clone());
    }

    return std::vector({ final_face_vec, final_face_not_vec, mask_img_not_vec, input_img_vec, probability_mask_vec, final_mask_vec, final_mask_not_vec });
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
        double fc = 15; // 50
        JointWMF filter;
        tmp1 = filter.filter(final_face[face], final_face[face], dx, fc);
        //bilateralFilter(final_face, tmp1, dx, fc, fc);

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
        cv::Mat probability_mask_resized, smoothed_resized;

        cv::Mat bgr_channels_smoothed[3], bgr_channels_orig[3];

        cv::split(original, bgr_channels_orig);

        for (int color = 0; color < 3; ++color)
        {
            cv::Mat original_color, smoothed_color;
            original_color = bgr_channels_orig[color];
            int width = original_color.rows;
            int height = original_color.cols;
            resize(smoothed[face], smoothed_resized, cv::Size(height, width), 0, 0, cv::INTER_CUBIC);
            cv::split(smoothed_resized.clone(), bgr_channels_smoothed);
            smoothed_color = bgr_channels_smoothed[color];

            resize(probability_masks[face], probability_mask_resized, cv::Size(height, width), 0, 0, cv::INTER_CUBIC);

            double coef = 0;

            uchar* original_color_ptr = original_color.data;
            uchar* smoothed_color_ptr = smoothed_color.data;
            uchar* probability_mask_resized_ptr = probability_mask_resized.data;

            parallel_for_(cv::Range(0, width * height), [&](const cv::Range& range) {
                for (size_t r = range.start; r < range.end; r++)
                {
                    double original_pixel = static_cast<double>(original_color_ptr[r]) / 255;
                    double smoothed_pixel = static_cast<double>(smoothed_color_ptr[r]) / 255;
                    double delta = static_cast<double>(abs(original_pixel - smoothed_pixel));
                    double intensity = delta;
                    double probability_pixel = static_cast<double>(probability_mask_resized_ptr[r]) / 255;
                    coef += intensity * probability_pixel;
                }
                });

            coef = ((gamma * coef) / 255) + beta;
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
                coefficients_HHi_smoothed_mat.emplace_back(cv::Mat(parts_original[i].row, parts_original[i].columns, CV_64F, getWT2Coeffs(wt_smoothed.get(), wavecoeffs_smoothed.get(), i + 1, 'D', &parts_original[i].columns, &parts_original[i].columns)));
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

cv::Mat Retouching(const cv::Mat& src, const std::string& model_dir) {
    std::vector < std::vector<cv::Mat> > masks_by_faces = MaskGenerate(src, model_dir);

    std::vector <cv::Mat> probability_masks = masks_by_faces[4];

    cv::Mat orig = masks_by_faces[3][0];

    std::vector <cv::Mat> smoothed = Smoothing(masks_by_faces);

    std::vector<std::vector<double>> coefs = CalculateCoef(probability_masks, orig, smoothed);

    std::vector<std::vector<cv::Mat>> faces(Restore(orig, smoothed, coefs));

    std::vector<cv::Mat> restored_faces, restored_faces_by_mask;
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
    imshow("mat", mat);
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