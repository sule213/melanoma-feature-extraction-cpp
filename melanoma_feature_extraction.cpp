
// FAST Corner Implementation

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/photo.hpp>
#include <stdio.h>
#include <iostream>
#include <string>
#include <map>

using namespace cv;
using namespace std;

void reduceColor_kmeans(const Mat3b& src, Mat3b& dst)
{
	int K = 8;
	int n = src.rows * src.cols;
	Mat data = src.reshape(1, n);
	data.convertTo(data, CV_32F);

	vector<int> labels;
	Mat1f colors;
	kmeans(data, K, labels, TermCriteria(), 1, KMEANS_PP_CENTERS, colors);

	for (int i = 0; i < n; ++i)
	{
		data.at<float>(i, 0) = colors(labels[i], 0);
		data.at<float>(i, 1) = colors(labels[i], 1);
		data.at<float>(i, 2) = colors(labels[i], 2);
	}

	Mat reduced = data.reshape(3, src.rows);
	reduced.convertTo(dst, CV_8U);
}

struct lessVec3b
{
	bool operator()(const Vec3b& lhs, const Vec3b& rhs) {
		return (lhs[0] != rhs[0]) ? (lhs[0] < rhs[0]) : ((lhs[1] != rhs[1]) ? (lhs[1] < rhs[1]) : (lhs[2] < rhs[2]));
	}
};

map<Vec3b, int, lessVec3b> getPalette(const Mat3b& src)
{
	map<Vec3b, int, lessVec3b> palette;
	for (int r = 0; r < src.rows; ++r)
	{
		for (int c = 0; c < src.cols; ++c)
		{
			Vec3b color = src(r, c);
			if (palette.count(color) == 0)
			{
				palette[color] = 1;
			}
			else
			{
				palette[color] = palette[color] + 1;
			}
		}
	}
	return palette;
}

Scalar Entropy(Mat image)
{

	vector<Mat> channels;
	split(image, channels);

	int histSize = 256;/// Establish the number of bins
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	Mat hist0, hist1, hist2;

	// Compute the histograms
	calcHist(&channels[0], 1, 0, Mat(), hist0, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&channels[1], 1, 0, Mat(), hist1, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&channels[2], 1, 0, Mat(), hist2, 1, &histSize, &histRange, uniform, accumulate);

	//frequency
	float f0 = 0, f1 = 0, f2 = 0;
	for (int i = 0; i<histSize; i++)
	{
		f0 += hist0.at<float>(i);
		f1 += hist1.at<float>(i);
		f2 += hist2.at<float>(i);
	}

	//entropy
	Scalar e;
	e.val[0] = 0;
	e.val[1] = 0;
	e.val[2] = 0;
	// e0=0, e1=0, e2=0;
	float p0, p1, p2;

	for (int i = 0; i<histSize; i++)
	{
		p0 = abs(hist0.at<float>(i)) / f0;
		p1 = abs(hist1.at<float>(i)) / f1;
		p2 = abs(hist2.at<float>(i)) / f2;
		if (p0 != 0)
			e.val[0] += -p0*log10(p0);
		if (p1 != 0)
			e.val[1] += -p1*log10(p1);
		if (p2 != 0)
			e.val[2] += -p2*log10(p2);
	}

	return e;
}

Scalar Skewness(Mat image)
{
	Scalar skewness, mean, stddev;
	skewness.val[0] = 0;
	skewness.val[1] = 0;
	skewness.val[2] = 0;
	meanStdDev(image, mean, stddev, Mat());
	int sum0, sum1, sum2;
	float den0 = 0, den1 = 0, den2 = 0;
	int N = image.rows*image.cols;

	for (int i = 0; i<image.rows; i++)
	{
		for (int j = 0; j<image.cols; j++)
		{
			sum0 = image.ptr<uchar>(i)[3 * j] - mean.val[0];
			sum1 = image.ptr<uchar>(i)[3 * j + 1] - mean.val[1];
			sum2 = image.ptr<uchar>(i)[3 * j + 2] - mean.val[2];

			skewness.val[0] += sum0*sum0*sum0;
			skewness.val[1] += sum1*sum1*sum1;
			skewness.val[2] += sum2*sum2*sum2;
			den0 += sum0*sum0;
			den1 += sum1*sum1;
			den2 += sum2*sum2;
		}
	}

	skewness.val[0] = skewness.val[0] * sqrt(N) / (den0*sqrt(den0));
	skewness.val[1] = skewness.val[1] * sqrt(N) / (den1*sqrt(den1));
	skewness.val[2] = skewness.val[2] * sqrt(N) / (den2*sqrt(den2));

	return skewness;
}

Scalar Kurtosis(Mat image)
{
	Scalar kurt, mean, stddev;
	kurt.val[0] = 0;
	kurt.val[1] = 0;
	kurt.val[2] = 0;
	meanStdDev(image, mean, stddev, Mat());
	int sum0, sum1, sum2;
	int N = image.rows*image.cols;
	float den0 = 0, den1 = 0, den2 = 0;

	for (int i = 0; i<image.rows; i++)
	{
		for (int j = 0; j<image.cols; j++)
		{
			sum0 = image.ptr<uchar>(i)[3 * j] - mean.val[0];
			sum1 = image.ptr<uchar>(i)[3 * j + 1] - mean.val[1];
			sum2 = image.ptr<uchar>(i)[3 * j + 2] - mean.val[2];

			kurt.val[0] += sum0*sum0*sum0*sum0;
			kurt.val[1] += sum1*sum1*sum1*sum1;
			kurt.val[2] += sum2*sum2*sum2*sum2;
			den0 += sum0*sum0;
			den1 += sum1*sum1;
			den2 += sum2*sum2;
		}
	}

	kurt.val[0] = (kurt.val[0] * N*(N + 1)*(N - 1) / (den0*den0*(N - 2)*(N - 3))) - (3 * (N - 1)*(N - 1) / ((N - 2)*(N - 3)));
	kurt.val[1] = (kurt.val[1] * N / (den1*den1)) - 3;
	kurt.val[2] = (kurt.val[2] * N / (den2*den2)) - 3;

	return kurt;
}

int main(int argv, char** argc)
{
	//string file_path = "images/malignant/002"; //cancerous
	string file_path = "images/benign/5"; //non cancerous

	Mat imageLoad, imageLab, imageEnhanced;
	imageLoad = imread(file_path + ".jpg");

	if (imageLoad.empty())
	{
		cout << "Image cannot be loaded..!!" << endl;
		return -1;
	}
	//Resize file
	resize(imageLoad, imageLoad, Size(1000, 750));

	//Enhance contrast by scale factor & delta added to the scaled values
	imageLoad.convertTo(imageLab, -1, 5, 2);

	//Apply Bone colormap to enhance the visualization
	applyColorMap(imageLab, imageLab, COLORMAP_BONE);

	//usually get better information from a HSV colorspace
	cvtColor(imageLab, imageEnhanced, CV_RGB2HSV);

	// define bounding rectangle 
	Rect rectangle(50, 80, imageEnhanced.cols - 150, imageEnhanced.rows - 180);

	Mat grabresult; // segmentation result (4 possible values)
	Mat bgModel, fgModel; // the models (internally used)

							  // GrabCut segmentation
	grabCut(imageEnhanced,    // input image
		grabresult,   // segmentation result
		rectangle,// rectangle containing foreground 
		bgModel, fgModel, // models
		1,        // number of iterations
		GC_INIT_WITH_RECT); // use rectangle

	// Get the pixels marked as likely foreground
	compare(grabresult, GC_PR_FGD, grabresult, CMP_EQ);

	// Generate output image
	Mat foreground(imageEnhanced.size(), CV_8UC3, Scalar(255, 255, 255));
	imageEnhanced.copyTo(foreground, grabresult); // bg pixels not copied

	// draw rectangle on original image
	cv::rectangle(imageEnhanced, rectangle, Scalar(255, 255, 255), 1);

	//Performs a mean-shift segmentation of the foreground image and eliminates small segments.
	int Spatial_window_radius = 20;
	int Color_window_radius = 20;
	Mat newforeground(imageEnhanced.size(), CV_8UC3);
	pyrMeanShiftFiltering(foreground, newforeground, Spatial_window_radius, Color_window_radius, 3);

	//smooth the image for segment edges
	Mat imageSmooth;
	int Max_Kernel_Length = 11;

	for (int i = 1; i < Max_Kernel_Length; i = i + 2)
	{
		//Median smooth the image in the "foreground" and save it to "dst"
		medianBlur(newforeground, imageSmooth, i);

		// Gaussian smoothing the image in the "foreground" and save it to "image2"
		//GaussianBlur(foreground, imageSmooth, Size(i,i), 0,0);
	}

	int startX = 20, startY = 0, width = 800, height = 700; //new image dimention

	Mat ROI(imageSmooth, Rect(startX, startY, width, height));//region for segmentation
	Mat ROI2(imageLoad, Rect(startX, startY, width, height));//region for masking

	Mat splitChannels[3];
	//spit image into 3 channels HSV

	split(ROI, splitChannels);

	//save Luminance channel[0] which skin color appers in and is also grayscale for further processing
	Mat gray, source;
	splitChannels[0].copyTo(source);

	//++++++Calculate Contour Area, Perimeter, Circularity, Aspect Ratio, Color Percentage++++++++

	Mat img_dialate;
	source.copyTo(img_dialate);

	//find contour
	threshold(img_dialate, img_dialate, 250, 255, THRESH_BINARY_INV); //Threshold the dialated

	//eliminate small blobs and noise using erode and dilate process
	erode(img_dialate, gray, Mat(), Point(-1, -1), 2, 1, 1);
	dilate(img_dialate, gray, (9, 9), Point(-1, -1), 2, 1, 1);

	//apply closing operator to fill hole gaps
	morphologyEx(img_dialate, gray, MORPH_CLOSE, Mat(), Point(-1, -1), 15);

	double largest_area = 2500, largest_perimeter = 0; //area and perimeter
	double largest_contour_index = 0, circularity = 0;//largest contour and circularity
	double convex_points = 0; int defect_count = 0; //number of defects
	double contour_area;
	
	//Rect bound_rect;
	vector<vector<Point>> contours; // Vector for storing contour
	vector<Vec4i> hierarchy;
	vector<int>small_blobs;

	// Find the contours in the image
	findContours(gray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	
	// Background image the same as the loaded image.
	Mat4b input_bgra, bkg = imread(file_path + ".jpg");

	// Create a mask with the inner part of the contour
	Mat4b mask2;
	drawContours(mask2, contours, 0, Scalar(255), CV_FILLED);

	// Create a white RGB image for background
	Mat4b bkgresult(ROI2.size(), Vec4b(255, 255, 255, 0));

	// Copy the image according to the mask
	bkg.copyTo(bkgresult, mask2);

	//convert the input image to BGRA channel (color image with alpha channel) for transparent
	cvtColor(mask2, input_bgra, CV_BGR2BGRA);

	// find all white pixel and set alpha value to zero:
	for (int y = 0; y < input_bgra.rows; ++y)
		for (int x = 0; x < input_bgra.cols; ++x)
		{
			cv::Vec4b & pixel = input_bgra.at<cv::Vec4b>(y, x);
			//if pixel is white
			if (pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255)
			{
				// set aplha to zero for calculating only non transparent region
				pixel[3] = 0;
			}
		}

	// Reduce color using k-means approach 
	Mat3b reduced;
	reduceColor_kmeans(bkgresult, reduced);

	// Get the different palette colors of image
	map<Vec3b, int, lessVec3b> palette = getPalette(reduced);

	// Print palette
	double area = bkgresult.rows * bkgresult.cols;
	int D_height = 50, D_width = 50;
	//Mat D (D_height, D_width, CV_32F);
	for (auto color : palette)
	{
		cout << "Color: " << color.first << " \t - Area: " << 100.f * float(color.second) / float(area) << "%" << endl;

		Mat D(D_height, D_width, CV_32F);
		D = cv::Scalar(color.first);
	}
	//Get the moments
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}
	//Get the mass centers:
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

		vector<double> total_res_of_area;
	Mat mask(ROI.cols, ROI.rows, CV_8UC1, Scalar::all(0));
	// iterate through each contour.
	for (int i = 0; i < contours.size(); i++)
	{
		//  Find the area of contour
		double area = contourArea(contours[i], false);
		total_res_of_area.push_back(area);

		//  Find the perimeter of contour
		double perimeter = arcLength(contours[i], false);

			if (area > largest_area) {
			largest_area = area; cout << i << " Area = " << area << endl;
			largest_perimeter = perimeter; cout << i << " Perimeter = " << perimeter << endl;
			circularity = (4 * CV_PI*area) / (perimeter*perimeter); cout << i << " Circularity = " << (4 * CV_PI*area) / (perimeter*perimeter) << endl;
			
			double equi_diameter = sqrt(4 * area / CV_PI); 

			// Store the index of largest contour
			largest_contour_index = i;
		}
	}

	/*vector<double> areas(contours.size());
	for (int i = 0; i < contours.size(); i++)
		areas[largest_contour_index] = contourArea(Mat(contours[largest_contour_index]));
	double max;
	Point maxPosition;
	minMaxLoc(Mat(areas), 0, &max, 0, &maxPosition);*/
	Mat source2;
	source.copyTo(source2);
	//drawContours(source2, contours,  maxPosition.y, Scalar(255, 255, 255), FILLED, 8, hierarchy);
	drawContours(source2, contours, largest_contour_index, Scalar(0, 0, 0), CV_FILLED, 8, hierarchy);

	//Get the corner points
	vector<Point2f> Corners;

	//good features to track color variation regions
	goodFeaturesToTrack(source, Corners, 200, 0.01, 20);
	double features = Corners.size();
	cout << "Features = " << features << endl;

	//mark feature areas 
	for (size_t i = 0; i < Corners.size(); i++)
	{
		circle(source, Corners[i], 10, Scalar(1), 4, 0);
	}

	//Convex hull obtained using convexHull() that should contain indices of the contour points that make the hull
	vector<vector<Point> >hull(contours.size());
	vector<vector<int> > hullsI(contours.size());
	vector<vector<Vec4i>> defects(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		convexHull(contours[i], hull[i], false);
		convexHull(contours[i], hullsI[i], false);
		if (hullsI[i].size() > 3)
		{
			convexityDefects(contours[i], hullsI[i], defects[i]);
		}
	}

	//contour is detected,then convex hull is found and also convexity defects are found and stored in defects
	if (area > largest_area) {
		drawContours(source, hull, largest_contour_index, Scalar(255, 0, 255), 3, 8, vector<Vec4i>(), 0, Point());

		// Draw convexityDefects
		for (int j = 0; j < defects[largest_contour_index].size(); ++j)
		{
			const Vec4i& v = defects[largest_contour_index][j];
			float depth = v[3] / 256;
			if (depth > 8) //  filter defects by depth

			{
				int start_idx = v[0]; Point ptStart(contours[largest_contour_index][start_idx]);
				int end_idx = v[1]; Point ptEnd(contours[largest_contour_index][end_idx]);
				int far_idx = v[2]; Point ptFar(contours[largest_contour_index][far_idx]);

				line(source, ptStart, ptEnd, Scalar(0, 255, 0), 1);
				line(source, ptStart, ptFar, Scalar(0, 255, 0), 1);
				line(source, ptEnd, ptFar, Scalar(0, 255, 0), 1);
				circle(source, ptFar, 4, Scalar(0, 255, 0), 2);

				defect_count++;
			}
		}
		cout << "Defects = " << defect_count << endl;
	}

	//Detects corners and Mask image
	Mat masking;
	int threshold = 100;
	vector<KeyPoint> keypoints;
	Scalar color(0, 255, 255);  // color of the contour in the

	//Detects corners using the FAST algorithm
	cv::FAST(gray, keypoints, threshold, true, FastFeatureDetector::TYPE_7_12);

	//Retain the specified number of the best keypoints 
	KeyPointsFilter::retainBest(keypoints, 200);

	//Remove duplicated keypoints
	KeyPointsFilter::removeDuplicated(keypoints);

	//Remove keypoints of sizes out of range
	KeyPointsFilter::runByKeypointSize(keypoints, 5);

	//Mark the corners
	drawKeypoints(source, keypoints, source, (0, 0, 255), 0);
	if (keypoints.size() > 0)
	{
		cout << "Number of Corners = " << keypoints.size() << endl;
	}

	//Draw the bounding rectangle, ellipse and minimum area rectangle
	Rect bounding_rect = boundingRect(contours[largest_contour_index]);

	vector<RotatedRect> minRect(contours.size());
	//vector<RotatedRect> minEllipse(contours.size());
	RotatedRect minEllipse;
	
		   for	(size_t i = 0; i < contours.size(); i++)
				{ minRect[largest_contour_index] = minAreaRect(Mat(contours[largest_contour_index]));
	       if	(contours[largest_contour_index].size() > 5)
		        //{ minEllipse[largest_contour_index] = fitEllipse(Mat(contours[largest_contour_index])); }
				{ minEllipse = fitEllipse(Mat(contours[largest_contour_index])); }
	       }
	
			for (size_t i = 0; i< contours.size(); i++)
		        {
		         Scalar color = Scalar(255,0,255);
		        // contour
			     drawContours(source, contours, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point());
		        // ellipse
			     //ellipse(source, minEllipse[largest_contour_index], color, 2, 8);
				 ellipse(source, minEllipse, color, 2, 8);
		        // rotated rectangle
			     Point2f rect_points[4]; minRect[largest_contour_index].points(rect_points);
		        for (int j = 0; j < 4; j++)
			     line(source, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
		     }

	//aspect ratio
	double contourwidth = bounding_rect.width;
	double contourheight = bounding_rect.height;
	double aspectRatio = contourwidth / contourheight; cout << "Aspect ratio = " << aspectRatio << endl;

			double max_diameter;
			if (contourwidth > contourheight)
				max_diameter = contourwidth;
			else
				max_diameter = contourheight;

			double Area = contourArea(contours[largest_contour_index], false);
			double Roundness = (4 * Area) / (CV_PI * max_diameter); cout << "Roundness = " << Roundness << endl;
			double Compactness = (sqrt((4 * Area) / CV_PI)) / max_diameter; cout << "Compactness = " << Compactness << endl;

	double ellipseArea = CV_PI*((minEllipse.size.width/2.0)*(minEllipse.size.height/2.0));
	cout << "Ellipse Area = " << ellipseArea << endl;

	double Extent = ellipseArea - Area; cout << "Extent = " << Extent << endl;

	//Calculate average pixel intensity
	Rect roi1(bounding_rect);
	Mat image_roi1 = ROI2(roi1);
	Scalar avgpixelIntensity1 = mean(image_roi1);
	double piori = avgpixelIntensity1.val[0];
	cout << "Pixel intensity of ROI = " << piori << endl;

	/*Rect roi2(bounding_rect);
	Mat image_roi2 = imageLab(roi2);
	Scalar avgpixelIntensity2 = mean(image_roi2);
	double piocm = avgpixelIntensity2.val[0];
	cout << "Pixel intensity of Color Map = " << piocm << endl;*/

	//Calculate average luminance
	Mat luminance;
	cvtColor(image_roi1, luminance, CV_BGR2YCrCb);
	Mat channels[3];
	split(luminance, channels);
	Mat lum = channels[2];
	Scalar avrglum = mean(lum);
	double lori = avrglum.val[0];
	cout << "Luminance over ROI = " << avrglum.val[0] << endl;

	//Calculate Entrophy
	Scalar ent; //0:1st channel, 1:2nd channel and 2:3rd channel
	ent = Entropy(image_roi1);
	cout << "Entropy : " << ent.val[0] << endl; 

	//Find Shape and Boundary Edge
	Mat edge, draw1, draw2;
	Canny(source2,edge,10,150,3);
	edge.convertTo(draw1,CV_8U);
	morphologyEx(draw1, draw1, MORPH_CLOSE, Mat(), Point(-1, -1), 5);
	draw1.copyTo(draw2);
	floodFill(draw2, Point(0, 0), Scalar(255));
	Mat im_floodfill_inv;
	bitwise_not(draw2, im_floodfill_inv);

	//Calculate Skewness
	Scalar skew_color, skew_shape, skew_bound; //0:1st channel, 1:2nd channel and 2:3rd channel
	skew_color = Skewness(image_roi1);
	skew_shape = Skewness(im_floodfill_inv);
	skew_bound = Skewness(draw1);
	cout << "Skewness Color : " << skew_color.val[0] << endl;
	cout << "Skewness Shape : " << skew_shape.val[0] << endl;
	cout << "Skewness Bound : " << skew_bound.val[0] << endl;

	//Calculate Kurtosis
	Scalar kurt_color, kurt_shape, kurt_bound; //0:1st channel, 1:2nd channel and 2:3rd channel
	kurt_color = Kurtosis(image_roi1);
	kurt_shape = Kurtosis(im_floodfill_inv);
	kurt_bound = Kurtosis(draw1);
	cout << "Kurtosis Color : " << kurt_color.val[0] << endl;
	cout << "Kurtosis Shape : " << kurt_shape.val[0] << endl;
	cout << "Kurtosis Bound : " << kurt_bound.val[0] << endl;

	//Variance
	Scalar mean, var;
	Mat var_mat;
	image_roi1.convertTo(var_mat, CV_32F, 1.0 / 255, 0);
	meanStdDev(var_mat, mean, var);
	//double V = var.val[0];
	cout << "Variance = " << var.val[0] << endl;

	//Draw the contours and mark center
	drawContours(gray, contours, largest_contour_index, color, CV_FILLED, 8, hierarchy);
	drawContours(source, contours, largest_contour_index, color, 1, 8, hierarchy);
	drawContours(mask, contours, largest_contour_index, color, CV_FILLED);
	circle(source, mc[largest_contour_index], 4, color, -1, 8, 0);
	putText(source, " center", mc[largest_contour_index], FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2);
	
	//Masking image
	ROI2.copyTo(masking, source);
	normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);

	cv::rectangle(gray, bounding_rect, color, 2, 8, 0);
	cv::rectangle(source, bounding_rect, color, 2, 8, 0);

	//output calculations on to result image 
	string temp_text1  = "Area := ";
	string temp_text2  = "Perimeter := ";
	string temp_text3  = "Circularity := ";
	string temp_text4  = "FAST Corners : = ";
	string temp_text5  = "Defects : = ";
	string temp_text6  = "Aspect ratio := ";
	string temp_text7  = "Features := ";
	string temp_text8  = "Avrg Pixel Intensity := ";
	string temp_text9  = "Avrg Luminance := ";
	string temp_text10 = "Entrophy := ";
	string temp_text11 = "Roundness := ";
	string temp_text12 = "Variance := ";
	string temp_text13 = "Skewness := ";
	string temp_text14 = "Kurtosis := ";
	string temp_text15 = "Extent := ";

	string temp_str1  = std::to_string(total_res_of_area[largest_contour_index]);
	string temp_str2  = std::to_string(largest_perimeter);
	string temp_str3  = std::to_string(circularity);
	string temp_str4  = std::to_string(keypoints.size());
	string temp_str5  = std::to_string(defect_count);
	string temp_str6  = std::to_string(aspectRatio);
	string temp_str7  = std::to_string(features);
	string temp_str8  = std::to_string(piori);
	string temp_str9  = std::to_string(lori);
	string temp_str10 = std::to_string(ent.val[0]);
	string temp_str11 = std::to_string(Roundness);
	string temp_str12 = std::to_string(var.val[0]);
	string temp_str13 = std::to_string(skew_color.val[0]) +", " + to_string(skew_shape.val[0]) + ", " + to_string(skew_bound.val[0]);
	string temp_str14 = std::to_string(kurt_color.val[0]) +", " + to_string(kurt_shape.val[0]) + ", " + to_string(kurt_bound.val[0]);
	string temp_str15 = std::to_string(Extent);

	cv::putText(source, temp_text1  + temp_str1, Point(20, 20), CV_FONT_NORMAL, 0.4f, Scalar(0, 0, 0), 1, 1, false);
	cv::putText(source, temp_text2  + temp_str2, Point(20, 40), CV_FONT_NORMAL, 0.4f, Scalar(0, 0, 0), 1, 1, false);
	cv::putText(source, temp_text3  + temp_str3, Point(20, 60), CV_FONT_NORMAL, 0.4f, Scalar(0, 0, 0), 1, 1, false);
	cv::putText(source, temp_text4  + temp_str4, Point(20, 80), CV_FONT_NORMAL, 0.4f, Scalar(0, 0, 0), 1, 1, false);
	cv::putText(source, temp_text5  + temp_str5, Point(20, 100), CV_FONT_NORMAL, 0.4f, Scalar(0, 0, 0), 1, 1, false);
	cv::putText(source, temp_text6  + temp_str6, Point(20, 120), CV_FONT_NORMAL, 0.4f, Scalar(0, 0, 0), 1, 1, false);
	cv::putText(source, temp_text7  + temp_str7, Point(20, 140), CV_FONT_NORMAL, 0.4f, Scalar(0, 0, 0), 1, 1, false);
	cv::putText(source, temp_text8  + temp_str8, Point(20, 160), CV_FONT_NORMAL, 0.4f, Scalar(0, 0, 0), 1, 1, false);
	cv::putText(source, temp_text9  + temp_str9, Point(20, 180), CV_FONT_NORMAL, 0.4f, Scalar(0, 0, 0), 1, 1, false);
	cv::putText(source, temp_text10 + temp_str10, Point(20, 200), CV_FONT_NORMAL, 0.4f, Scalar(0, 0, 0), 1, 1, false);
	cv::putText(source, temp_text11 + temp_str11, Point(20, 220), CV_FONT_NORMAL, 0.4f, Scalar(0, 0, 0), 1, 1, false);
	cv::putText(source, temp_text12 + temp_str12, Point(20, 240), CV_FONT_NORMAL, 0.4f, Scalar(0, 0, 0), 1, 1, false);
	cv::putText(source, temp_text13 + temp_str13, Point(20, 260), CV_FONT_NORMAL, 0.4f, Scalar(0, 0, 0), 1, 1, false);
	cv::putText(source, temp_text14 + temp_str14, Point(20, 280), CV_FONT_NORMAL, 0.4f, Scalar(0, 0, 0), 1, 1, false);
	cv::putText(source, temp_text15 + temp_str15, Point(20, 300), CV_FONT_NORMAL, 0.4f, Scalar(0, 0, 0), 1, 1, false);

	namedWindow("Display window", CV_WINDOW_AUTOSIZE);
	imshow("Display window", source);
	namedWindow("Output", CV_WINDOW_AUTOSIZE);
	imshow("Output", masking);
	namedWindow("Mask", CV_WINDOW_AUTOSIZE);
	imshow("Mask", imageSmooth);
	namedWindow("inverse", CV_WINDOW_AUTOSIZE);
	imshow("inverse", im_floodfill_inv);
	namedWindow("canny", CV_WINDOW_AUTOSIZE);
	imshow("canny", draw1);
	imwrite(file_path+"results.png", source);
	imwrite(file_path+"masked.png", masking);
	//imwrite(file_path + "colorpalette.png", D);

	waitKey(100000);
	return 0;
}
