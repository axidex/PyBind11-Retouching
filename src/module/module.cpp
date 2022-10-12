#include <pybind11/pybind11.h>
#include "ndarray_converter.h"
//#include <pybind11/numpy.h>

//#include"retouching.cpp"

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
#include <newfilter.cpp>

#include <cassert>
#include <fstream>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include <algorithm>
#include <memory>


namespace py = pybind11;


// debugging libs
//#include <chrono>
//#include <typeinfo>

std::vector<cv::Mat> maskGenerate(cv::Mat src, std::string modelDir)
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
    const auto drawLandmark = [&](const auto x, const auto y) {
        constexpr auto radius = 5;
        const auto color = cv::Scalar(0, 255, 255);
        constexpr auto thickness = 2;
        const auto center = cv::Point(x, y);
        cv::circle(landmarkImg, center, radius, color, thickness);
    };

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
        // Join the landmark points on the boundary of facial features using cubic curve

        const auto getCurve = [&](int n, int nend) {
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
        };

        // Right eye cubic curve
        constexpr auto nEyeCurve = 6;
        const auto rightEyeCurve = getCurve(36, 41);

        // Sample landmark points from the curve
        constexpr auto eyePointNum = 25;
        std::array<cv::Point, eyePointNum> rightEyePts;
        for (int i = 0; i < eyePointNum; ++i) {
            const auto net = rightEyeCurve(1.0 / eyePointNum * i);
            const auto result = net.result();
            const auto x = result[0], y = result[1];
            drawLandmark(x, y);
            rightEyePts[i] = cv::Point(x, y);
        }
        // Draw binary mask
        cv::fillConvexPoly(maskImg, rightEyePts, cv::Scalar(255), cv::LINE_AA);

        // Left eye cubic curve
        const auto leftEyeCurve = getCurve(42, 47);
        std::array<cv::Point, eyePointNum> leftEyePts;
        // Sample landmark points from the curve
        for (int i = 0; i < eyePointNum; ++i) {
            const auto net = leftEyeCurve(1.0 / eyePointNum * i);
            const auto result = net.result();
            const auto x = result[0], y = result[1];
            drawLandmark(x, y);
            leftEyePts[i] = cv::Point(x, y);
        }
        // Draw binary mask
        cv::fillConvexPoly(maskImg, leftEyePts, cv::Scalar(255), cv::LINE_AA);

        // Mouth cubic curve
        constexpr auto nMouthCurve = 12;
        const auto mouthCurve = getCurve(48, 59);
        constexpr auto mouthPointNum = 40;
        std::array<cv::Point, mouthPointNum> mouthPts;
        // Sample landmark points from the curve
        for (int i = 0; i < mouthPointNum; ++i) {
            const auto net = mouthCurve(1.0 / mouthPointNum * i);
            const auto result = net.result();
            const auto x = result[0], y = result[1];
            drawLandmark(x, y);
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
            drawLandmark(x, y);
            lowerFacePts.push_back(cv::Point(x, y));
        }
        // Guess a point located in the upper face region
        // Pb: 8 (bottom of jaw)
        // Pt: 27 (top of nose
        const auto& Pb = shape.part(8);
        const auto& Pt = shape.part(27);
        const auto x = Pb.x();
        const auto y = Pt.y() - 0.85 * abs(Pb.y() - Pt.y());
        drawLandmark(x, y);
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

    cv::imwrite("maskImg3C.jpg", maskImg3C);
    cv::ximgproc::guidedFilter(spotImgT, maskImg3C, maskGF, 10, 200); //10 200
    cv::bitwise_not(maskGF, maskImgNot);

    cv::bitwise_and(inputImg, maskGF, spotImg);
    cv::Mat t;


    cv::Mat tmp1, tmp2;

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
    cv::bitwise_not(final_mask, final_mask_not);
    cv::Mat final_face_not, final_face;
    cv::bitwise_and(workImg, final_mask, final_face);
    cv::bitwise_and(workImg, final_mask_not, final_face_not);

    return std::vector({ final_face, final_face_not, maskImgNot, inputImg });
}
std::vector<cv::Mat> maskGenerate(std::string imgDir, std::string modelDir)
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

cv::Mat smoothing(cv::Mat final_face, cv::Mat final_face_not, cv::Mat maskImgNot, cv::Mat inputImg)
{
    cv::Mat tmp1, tmp2;
    cv::Mat noFace;
    int dx = 4; // 5
    double fc = 10; // 50
    JointWMF filter;
    tmp1 = filter.filter(final_face, final_face, dx, fc);
    //bilateralFilter(final_face, tmp1, dx, fc, fc);

    cv::bitwise_and(inputImg, maskImgNot, noFace);

    cv::add(final_face_not, tmp1, tmp2);

    //dst = filter.filter(tmp2, tmp2, 2, 10);
    //bilateralFilter(tmp2, dst, 5, 20, 20);
    return tmp2.clone();
}
cv::Mat smoothing(std::vector<cv::Mat> masks)
{
    return smoothing(masks[0], masks[1], masks[2], masks[3]);
}

std::vector<cv::Mat> restore(cv::Mat orig, cv::Mat smoothed, int alfa, int beta)
{
    cv::Mat bgrchannel_smoothed[3], bgrchannel_orig[3];

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
        wave_object obj_smoothed, obj_orig;
        wt2_object wt_smoothed, wt_orig;

        double* wavecoeffs_smoothed, * wavecoeffs_orig;
        double* cHH1_smoothed, * cHH2_smoothed, * cHH3_smoothed, * cHH1_orig, * cHH2_orig, * cHH3_orig;
        int i1r, i1c, i2r, i2c, i3r, i3c;
        int is1r, is1c, is2r, is2c, is3r, is3c;

        const char* name = "db2";

        obj_smoothed = wave_init(name);
        obj_orig = wave_init(name);

        wt_orig = wt2_init(obj_orig, "dwt", orig.rows, orig.cols, J);
        wt_smoothed = wt2_init(obj_smoothed, "dwt", smoothed.rows, smoothed.cols, J);

        wavecoeffs_orig = dwt2(wt_orig, color_orig);
        wavecoeffs_smoothed = dwt2(wt_smoothed, color_smoothed);

        cHH1_orig = getWT2Coeffs(wt_orig, wavecoeffs_orig, 1, 'D', &i1r, &i1c);
        cHH2_orig = getWT2Coeffs(wt_orig, wavecoeffs_orig, 2, 'D', &i2r, &i2c);
        cHH3_orig = getWT2Coeffs(wt_orig, wavecoeffs_orig, 3, 'D', &i3r, &i3c);

        cHH1_smoothed = getWT2Coeffs(wt_smoothed, wavecoeffs_smoothed, 1, 'D', &is1r, &is1c);
        cHH2_smoothed = getWT2Coeffs(wt_smoothed, wavecoeffs_smoothed, 2, 'D', &is2r, &is2c);
        cHH3_smoothed = getWT2Coeffs(wt_smoothed, wavecoeffs_smoothed, 3, 'D', &is3r, &is3c);

        //dispWT2Coeffs(cHH1s, i1r, i1c);
        //std::cout << is1r << " " << is1c << std::endl;
        //std::cout << i1r << " " << i2r << " " << i3r << " " << i1c << " " << i2c << " " << i3c << " " << std::endl;
        //std::cout << is1r << " " << is2r << " " << is3r << " " << is1c << " " << is2c << " " << is3c << " " << std::endl;
        int rows = i1r;
        int cols = i1c;

        cv::Mat cHH1_orig_mat(i1r, i1c, CV_64F, cHH1_orig);
        cv::Mat cHH2_orig_mat(i2r, i2c, CV_64F, cHH2_orig);
        cv::Mat cHH3_orig_mat(i3r, i3c, CV_64F, cHH3_orig);

        cv::Mat cHH1_smoothed_mat(i1r, i1c, CV_64F, cHH1_smoothed);
        cv::Mat cHH2_smoothed_mat(i2r, i2c, CV_64F, cHH2_smoothed);
        cv::Mat cHH3_smoothed_mat(i3r, i3c, CV_64F, cHH3_smoothed);

        cv::addWeighted(cHH1_orig_mat, alfa, cHH1_smoothed_mat, beta, 0, cHH1_smoothed_mat);
        cv::addWeighted(cHH2_orig_mat, alfa, cHH2_smoothed_mat, beta, 0, cHH2_smoothed_mat);
        cv::addWeighted(cHH3_orig_mat, alfa, cHH3_smoothed_mat, beta, 0, cHH3_smoothed_mat);

        //mid(cHH1_orig, cHH1_smoothed, is1r, is1c);
        //mid(cHH2_orig, cHH2_smoothed, is2r, is2c);
        //mid(cHH3_orig, cHH3_smoothed, is3r, is3c);
        cv::Mat oupMat = cv::Mat::zeros(smoothed.rows, smoothed.cols, CV_64F);

        double* oup = oupMat.ptr<double>(0);

        idwt2(wt_smoothed, wavecoeffs_smoothed, oup);

        colors.push_back(oupMat);
        //std::cout << color << std::endl;
        wave_free(obj_orig);
        wt2_free(wt_orig);
        free(wavecoeffs_orig);
        free(wavecoeffs_smoothed);
    }
    //cv::Mat final_eachCh[3] = {colors[0], colors[1], colors[3]};
    cv::Mat convertedMat_blue, convertedMat_green, convertedMat_red;
    colors[0].convertTo(convertedMat_blue, CV_8U);
    colors[1].convertTo(convertedMat_green, CV_8U);
    colors[2].convertTo(convertedMat_red, CV_8U);

    return std::vector({ convertedMat_blue, convertedMat_green, convertedMat_red });
}

cv::Mat Retouching(cv::Mat src) {

    std::string modelDir = "shape_predictor_68_face_landmarks.dat";

    std::vector<cv::Mat> masks = maskGenerate(src, modelDir);

    cv::Mat orig = masks[3];

    cv::Mat smoothed = smoothing(masks);

    double alfa, beta;
    alfa = 0.9;
    beta = 1 - alfa;

    std::vector<cv::Mat> colors = restore(orig, smoothed, alfa, beta);

    cv::Mat final_eachCh[3] = { colors[0], colors[1], colors[2] };
    cv::Mat final_colors;
    cv::merge(final_eachCh, 3, final_colors);

    return final_colors;
}

cv::Mat RetouchingImg(std::string imgDir) {

    std::string modelDir = "shape_predictor_68_face_landmarks.dat";

    const auto inputImg =
        cv::imread(cv::samples::findFile(imgDir, /*required=*/false, /*silentMode=*/true));
    if (inputImg.empty()) {
        std::cout << "Could not open or find the image: " << imgDir << "\n"
            << "The image should be located in `images_dir`.\n";
        assert(false);
    }
    return Retouching(inputImg);
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

	module.def("Retouching", &Retouching, "A function that return the matrix of Retouched image", py::arg("src"));
    module.def("RetouchingImg", &RetouchingImg, "A function that return the matrix of Retouched image", py::arg("imgDir"));
    module.def("Show", &Show, "imshow", py::arg("mat"));
}
