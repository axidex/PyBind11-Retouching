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
#include <iostream>
#include <newfilter.hpp>

#include <cassert>
#include <fstream>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include <algorithm>
#include <memory>
#include <functional>

namespace py = pybind11;

// debugging libs
//#include <chrono>
//#include <typeinfo>

tinyspline::BSpline getCurve(int n, int nend, std::vector<tinyspline::real>& knots, const dlib::full_object_detection& shape)
{
    knots.clear();

    for (int i = n; i <= nend; ++i) {
        const auto& point = shape.part(i);
        knots.push_back(point.x());
        knots.push_back(point.y());
    }

    // Make a closed curve
    knots.push_back(knots[0]);
    knots.push_back(knots[1]);
    // Interpolate the curve
    auto spline = tinyspline::BSpline::interpolateCubicNatural(knots, 2);

    return spline;
}

void drawLandmark(double x, double y, cv::Mat& landmarkImg)
{
    constexpr auto radius = 5;
    const auto color = cv::Scalar(0, 255, 255);
    constexpr auto thickness = 2;
    const auto center = cv::Point(x, y);
    cv::circle(landmarkImg, center, radius, color, thickness);
}

void face_part( cv::Mat& maskImg, const dlib::shape_predictor landmarkDetector, const dlib::full_object_detection shape, int entry, int end, cv::Mat& landmarkImg)
{
    std::vector<tinyspline::real> knots;

    // Right eye cubic curve
    const auto Curve = getCurve(entry, end, knots, shape);
    // Sample landmark points from the curve
    std::array<cv::Point, size_t(25)> Pts;
    for (int i = 0; i < 25; ++i) {
        const auto net = Curve(1.0 / 25 * i);
        const auto result = net.result();
        const auto x = result[0], y = result[1];
        drawLandmark(x, y, landmarkImg);
        Pts[i] = cv::Point(x, y);
    }
    // Draw binary mask
    cv::fillConvexPoly(maskImg, Pts, cv::Scalar(255), cv::LINE_AA);
}

std::vector<cv::Mat> maskGenerate(const cv::Mat& src,const string& modelDir)
{
    const auto inputImg = src;
    // Make a copy for drawing landmarks
    cv::Mat landmarkImg = inputImg.clone();
    // Make a copy for drawing binary mask
    cv::Mat maskImg = cv::Mat::zeros(inputImg.size(), CV_8UC1);


    auto landmarkModelPath = cv::samples::findFile(modelDir, /*required=*/false);
    if (landmarkModelPath.empty()) {
        std::cout << "Could not find the landmark model file: " << modelDir << "\n"
            << "The model should be located in `models_dir`.\n";
        assert(false);
    }

    // Leave the original input image untouched
    cv::Mat workImg = inputImg.clone();

    dlib::shape_predictor landmarkDetector;
    dlib::deserialize(landmarkModelPath) >> landmarkDetector;

    // Detect faces
    // Need to use `dlib::cv_image` to bridge OpenCV and dlib.
    const auto dlibImg = dlib::cv_image<dlib::bgr_pixel>(inputImg);
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

    for (const auto& face : faces) {
        std::vector<tinyspline::real> knots;
        const auto shape = landmarkDetector(dlibImg, face);

        face_part(maskImg, landmarkDetector, shape, 36, 41, landmarkImg); // right eye
        face_part(maskImg, landmarkDetector, shape, 42, 47, landmarkImg); // left eye
        //face_part(maskImg, landmarkDetector, shape, 48, 59, landmarkImg);

        const auto mouthCurve = getCurve(48, 59, knots, shape);
        constexpr auto mouthPointNum = 40;
        std::array<cv::Point, mouthPointNum> mouthPts;
        // Sample landmark points from the curve
        for (int i = 0; i < mouthPointNum; ++i) {
            const auto net = mouthCurve(1.0 / mouthPointNum * i);
            const auto result = net.result();
            const auto x = result[0], y = result[1];
            drawLandmark(x, y, landmarkImg);
            mouthPts[i] = cv::Point(x, y);
        }
        // Draw binary mask
        cv::fillPoly(maskImg, mouthPts, cv::Scalar(255), cv::LINE_AA);
        // Estimate an ellipse that can complete the upper face region
        constexpr auto nJaw = 17;
        std::vector<cv::Point> lowerFacePts;
        for (int i = 0; i < nJaw; ++i) {
            const auto& point = shape.part(i);
            const auto x = point.x(), y = point.y();
            drawLandmark(x, y, landmarkImg);
            lowerFacePts.push_back(cv::Point(x, y));
        }
        // Guess a point located in the upper face region
        // Pb: 8 (bottom of jaw)
        // Pt: 27 (top of nose
        const auto& Pb = shape.part(8);
        const auto& Pt = shape.part(27);
        const auto x = Pb.x();
        const auto y = Pt.y() - 0.85 * abs(Pb.y() - Pt.y());
        drawLandmark(x, y, landmarkImg);
        lowerFacePts.push_back(cv::Point(x, y));
        // Fit ellipse
        const auto box = cv::fitEllipseDirect(lowerFacePts);
        cv::Mat maskTmp = cv::Mat(maskImg.size(), CV_8UC1, cv::Scalar(255));
        cv::ellipse(maskTmp, box, cv::Scalar(0), /*thickness=*/-1, cv::FILLED);

        cv::bitwise_or(maskTmp, maskImg, maskImg);
        cv::bitwise_not(maskImg, maskImg);
    }

    cv::Mat maskChannels[3] = { maskImg, maskImg, maskImg };
    cv::Mat maskImg3C;
    cv::merge(maskChannels, 3, maskImg3C);
    cv::Mat spotImg, spotImgT;
    cv::Mat maskImgNot, maskGF;

    cv::bitwise_and(inputImg, maskImg3C, spotImgT);

    cv::ximgproc::guidedFilter(spotImgT, maskImg3C, maskGF, 10, 200); //10 200
    cv::bitwise_not(maskGF, maskImgNot);

    cv::bitwise_and(inputImg, maskGF, spotImg);

    // Inner mask
    cv::Mat maskEx;
    cv::Mat maskElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(30, 30)); //71 71

    cv::morphologyEx(maskImg, maskEx, cv::MORPH_ERODE, maskElement);
    cv::Mat maskExs[3] = { maskEx, maskEx, maskEx };
    cv::Mat maskEx3C;
    cv::merge(maskExs, 3, maskEx3C);

    // Make a preserved image for future use
    cv::Mat preservedImg, maskPres;
    cv::bitwise_not(maskEx3C, maskPres);
    cv::bitwise_and(workImg, maskPres, preservedImg);

    // Spot Concealment
    // Convert the RGB image to a single channel gray image
    cv::Mat grayImg;
    cv::cvtColor(workImg, grayImg, cv::COLOR_BGR2GRAY);

    // Compute the DoG to detect edges
    cv::Mat blurImg1, blurImg2, dogImg;
    const auto sigmaY = grayImg.cols / 200.0;
    const auto sigmaX = grayImg.rows / 200.0;
    cv::GaussianBlur(grayImg, blurImg1, cv::Size(3, 3), /*sigma=*/0);
    cv::GaussianBlur(grayImg, blurImg2, cv::Size(0, 0), sigmaX, sigmaY);
    cv::subtract(blurImg2, blurImg1, dogImg);
    cv::Mat not_dogImg;
    cv::bitwise_not(dogImg, not_dogImg);

    // Apply binary mask to the image
    cv::Mat not_dogImgs[3] = { not_dogImg, not_dogImg, not_dogImg };
    cv::Mat not_dogImg3C;
    cv::merge(not_dogImgs, 3, not_dogImg3C);

    cv::Mat final_mask, final_mask_not;

    cv::bitwise_and(maskGF, not_dogImg3C, final_mask);
    
    cv::threshold(final_mask, final_mask, 230, 255, cv::THRESH_BINARY);

    cv::bitwise_not(final_mask, final_mask_not);
    cv::Mat final_face_not, final_face;
    cv::bitwise_and(workImg, final_mask, final_face);
    cv::bitwise_and(workImg, final_mask_not, final_face_not);

    return std::vector({ final_face, final_face_not, maskImgNot, inputImg });
}
std::vector<cv::Mat> maskGenerate(const std::string& imgDir, const std::string& modelDir)
{
    const auto inputImg =
        cv::imread(cv::samples::findFile(imgDir, /*required=*/false, /*silentMode=*/true));
    if (inputImg.empty()) {
        std::cout << "Could not open or find the image: " << imgDir << "\n"
            << "The image should be located in `images_dir`.\n";
        assert(false);
    }
    return maskGenerate(inputImg, modelDir);
}

cv::Mat smoothing(cv::Mat& final_face, const cv::Mat& final_face_not,const cv::Mat& maskImgNot,const cv::Mat& inputImg)
{
    cv::Mat tmp1, tmp2;
    cv::Mat noFace;
    int dx = 5; // 5
    double fc = 15; // 50
    JointWMF filter;
    tmp1 = filter.filter(final_face, final_face, dx, fc);
    //bilateralFilter(final_face, tmp1, dx, fc, fc);

    cv::bitwise_and(inputImg, maskImgNot, noFace);

    cv::add(final_face_not, tmp1, tmp2);

    //dst = filter.filter(tmp2, tmp2, 2, 10);
    //bilateralFilter(tmp2, dst, 5, 20, 20);
    return tmp2.clone();
}
cv::Mat smoothing(std::vector<cv::Mat>& masks)
{
    return smoothing(masks[0], masks[1], masks[2], masks[3]);
}

struct WaveDeleter {
    void operator()(wave_set* b) { wave_free(b); }
};

template<typename T>
using deleted_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

std::vector<cv::Mat> restore(const cv::Mat& orig,const cv::Mat& smoothed, double alfa)
{
    cv::Mat bgrchannel_smoothed[3], bgrchannel_orig[3];
    assert(!alfa <= 1 || !alfa >= 0);
    double beta = 1 - alfa;
    cv::split(smoothed.clone(), bgrchannel_smoothed);
    cv::split(orig.clone(), bgrchannel_orig);
    int J = 3;
    cv::Mat double_smoothed, double_orig;
    int N = smoothed.rows * smoothed.cols;
    std::vector<cv::Mat> colors;
    for (int color = 0; color < 3; ++color)
    {
        bgrchannel_smoothed[color].convertTo(double_smoothed, CV_64F);
        bgrchannel_orig[color].convertTo(double_orig, CV_64F);
        double* color_smoothed = double_smoothed.ptr<double>(0);
        double* color_orig = double_orig.ptr<double>(0);

        const char* name = "db2";
        deleted_unique_ptr<wave_set> obj_smoothed(wave_init(name), [](wave_set* f) { wave_free(f); });
        deleted_unique_ptr<wave_set> obj_orig(wave_init(name), [](wave_set* f) { wave_free(f); });

        deleted_unique_ptr<wt2_set> wt_smoothed(wt2_init(obj_smoothed.get(), "dwt", smoothed.rows, smoothed.cols, J), [](wt2_set* f) { wt2_free(f); });
        deleted_unique_ptr<wt2_set> wt_orig(wt2_init(obj_orig.get(), "dwt", orig.rows, orig.cols, J), [](wt2_set* f) { wt2_free(f); });

        constexpr size_t kTotalParts = 3;
            struct {
            int row;
            int columns;
        } parts_smoothed[kTotalParts], parts_original[kTotalParts];

        deleted_unique_ptr<double> wavecoeffs_orig( dwt2(wt_orig.get(), color_orig), [](double* f) { free(f); });
        deleted_unique_ptr<double> wavecoeffs_smoothed( dwt2(wt_smoothed.get(), color_smoothed), [](double* f) { free(f); });

        cv::Mat cHH1_orig_mat(parts_original[0].row, parts_original[0].columns, CV_64F, getWT2Coeffs(wt_orig.get(), wavecoeffs_orig.get(), 1, 'D', &parts_original[0].row, &parts_original[0].columns));
        cv::Mat cHH2_orig_mat(parts_original[1].row, parts_original[1].columns, CV_64F, getWT2Coeffs(wt_orig.get(), wavecoeffs_orig.get(), 2, 'D', &parts_original[1].row, &parts_original[1].columns));
        cv::Mat cHH3_orig_mat(parts_original[2].row, parts_original[2].columns, CV_64F, getWT2Coeffs(wt_orig.get(), wavecoeffs_orig.get(), 3, 'D', &parts_original[2].row, &parts_original[2].columns));

        cv::Mat cHH1_smoothed_mat(parts_original[0].row, parts_original[0].columns, CV_64F, getWT2Coeffs(wt_smoothed.get(), wavecoeffs_smoothed.get(), 1, 'D', &parts_original[0].columns, &parts_original[0].columns));
        cv::Mat cHH2_smoothed_mat(parts_original[1].row, parts_original[1].columns, CV_64F, getWT2Coeffs(wt_smoothed.get(), wavecoeffs_smoothed.get(), 2, 'D', &parts_original[1].columns, &parts_original[1].columns));
        cv::Mat cHH3_smoothed_mat(parts_original[2].row, parts_original[2].columns, CV_64F, getWT2Coeffs(wt_smoothed.get(), wavecoeffs_smoothed.get(), 3, 'D', &parts_original[2].columns, &parts_original[2].columns));

        cv::addWeighted(cHH1_orig_mat, alfa, cHH1_smoothed_mat, beta, 0, cHH1_smoothed_mat);
        cv::addWeighted(cHH2_orig_mat, alfa, cHH2_smoothed_mat, beta, 0, cHH2_smoothed_mat);
        cv::addWeighted(cHH3_orig_mat, alfa, cHH3_smoothed_mat, beta, 0, cHH3_smoothed_mat);

        cv::Mat oupMat = cv::Mat::zeros(smoothed.rows, smoothed.cols, CV_64F);

        double* oup = oupMat.ptr<double>(0);

        idwt2(wt_smoothed.get(), wavecoeffs_smoothed.get(), oup);

        colors.push_back(oupMat);
    }
    cv::Mat convertedMat_blue, convertedMat_green, convertedMat_red;
    colors[0].convertTo(convertedMat_blue, CV_8U);
    colors[1].convertTo(convertedMat_green, CV_8U);
    colors[2].convertTo(convertedMat_red, CV_8U);

    return std::vector({ convertedMat_blue, convertedMat_green, convertedMat_red });
}

cv::Mat Retouching(const cv::Mat& src, const std::string& modelDir, double alfa) {

    

    std::vector<cv::Mat> masks = maskGenerate(src, modelDir);

    cv::Mat orig = masks[3];

    cv::Mat smoothed = smoothing(masks);

    std::vector<cv::Mat> colors = restore(orig, smoothed, alfa);

    cv::Mat final_eachCh[3] = { colors[0], colors[1], colors[2] };
    cv::Mat final_colors;
    cv::merge(final_eachCh, 3, final_colors);

    return final_colors;
}

cv::Mat RetouchingImg(const std::string& imgDir, const std::string& modelDir, double alfa) {

    const auto inputImg =
        cv::imread(cv::samples::findFile(imgDir, /*required=*/false, /*silentMode=*/true));
    if (inputImg.empty()) {
        std::cout << "Could not open or find the image: " << imgDir << "\n"
            << "The image should be located in `images_dir`.\n";
        assert(false);
    }
    return Retouching(inputImg, modelDir,alfa);
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

	module.def("Retouching", &Retouching, "Retouching function\n\nArguments:\n\tsrc : array like\n\tinput image\nmodelDir : string\n\t path to the your model\nalfa : float\n\tcoefficient for wavelet transform\n\nReturns\noutput :array like\n\toutput image"
        , py::arg("src"), py::arg("modelDir"), py::arg("alfa"));
    module.def("RetouchingImg", &RetouchingImg, "Retouching function\n\nArguments:\n\timgDir : string\n\tpath to input image\nmodelDir : string\n\t path to the your model\nalfa : float\n\tcoefficient for wavelet transform\n\nReturns\noutput : array like\n\toutput image"
        , py::arg("imgDir"), py::arg("modelDir"), py::arg("alfa"));
    module.def("Show", &Show, "imshow", py::arg("mat"));
}