#include <iostream>
#include<opencv2/core/core.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include"opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <stack>



using namespace std;
using namespace cv;

//==============================自动阈值分割==============================//
int myOtsu(Mat img) {
	if (img.channels() != 1) {
		cerr << "please input the gray picture" << endl;
	}

	float pixel_porb[256] = { 0 };
	for (int x = 0; x < img.rows; x++) {
		for (int y = 0; y < img.cols; y++) {
			pixel_porb[img.at<uchar>(x, y)]++;
		}
	}
	for (int i = 0; i < 256; i++) {
		pixel_porb[i] /= img.rows * img.cols;
	}

	float gmax = 0;
	int threshould = 0;
	for (int i = 0; i < 256; i++) {
		float w0 = 0;
		float w1 = 0;
		float u0 = 0;
		float u1 = 0;
		for (int j = 0; j < 256; j++) {
			if (i <= j) {
				w0 += pixel_porb[j];
				u0 += j * pixel_porb[j];
			}
			else {
				w1 += pixel_porb[j];
				u1 += j * pixel_porb[j];
			}
		}
		//平均灰度
		float u = u0 + u1;
		u0 /= w0;
		u1 /= w1;
		float g = w0 * pow((u - u0), 2) + w1 * pow((u - u1), 2);
		if (g > gmax) {
			gmax = g;
			threshould = i;
		}
	}
	return threshould;
}

Mat mythreshould(Mat img, int threshould) {
	if (img.channels() != 1) {
		cerr << "please input the gray picture" << endl;
	}

	Mat bw(img.rows, img.cols, img.type());
	for (int x = 0; x < img.rows; x++) {
		for (int y = 0; y < img.cols; y++) {
			if (img.at<uchar>(x, y) <= threshould) {
				bw.at<uchar>(x, y) = 255;
			}
			else {
				bw.at<uchar>(x, y) = 0;
			}
		}
	}
	return bw;
}

//========================================================================//
//种子填充法
Mat countRegion(Mat bw) {
	Mat mask(bw.size(), CV_16U);
	//修改为0-1二值图像
	bw.convertTo(mask, CV_16U);
	mask /= 255;
	//标签从2开始计算，因为可能会有多余256个连通区域
	int label = 2;
	for (int x = 1; x < bw.rows - 1; x++) {
		for (int y = 1; y < bw.cols - 1; y++) {
			if (mask.at<uint16_t>(x, y) == 1) {
				//建立坐标存贮栈
				stack<pair<int, int>> neighborPixels;
				//将当前像素值为255的坐标存入栈中
				neighborPixels.push(pair<int, int>(x, y));
				label++;
				//当栈不为空时，表明还有连通区域像素存在
				while(!neighborPixels.empty()) {
					//获得当前像素坐标
					pair<int, int> nowPixel = neighborPixels.top();
					int now_x = nowPixel.first;
					int now_y = nowPixel.second;
					//当前坐标出栈
					neighborPixels.pop();
					//赋予标签
					mask.at<uint16_t>(now_x, now_y) = label;
					//四邻域入栈判断
					if (mask.at<uint16_t>(now_x - 1, now_y) == 1) {
						neighborPixels.push(pair<int, int>(now_x - 1, now_y));
					}
					if (mask.at<uint16_t>(now_x + 1, now_y) == 1) {
						neighborPixels.push(pair<int, int>(now_x + 1, now_y));
					}
					if (mask.at<uint16_t>(now_x, now_y - 1) == 1) {
						neighborPixels.push(pair<int, int>(now_x, now_y - 1));
					}
					if (mask.at<uint16_t>(now_x, now_y + 1) == 1) {
						neighborPixels.push(pair<int, int>(now_x, now_y + 1));
					}
				}
			}
		}
	}

	return mask;
}

int main() {
	Mat img = imread("C://Users//Chrysanthemum//Desktop//0.png",0);

	int myOtsuThreshould = myOtsu(img);
	Mat bw = mythreshould(img, myOtsuThreshould);

	Mat mask = countRegion(bw);
	//显示，使得结果看的清晰，因为标签是从2开始的，像素值为2，看不清的
	imshow("count", mask*5000);

	imshow("origin pic", img);
	imshow("bw pic", bw);
	waitKey();
}