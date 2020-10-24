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

//==============================�Զ���ֵ�ָ�==============================//
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
		//ƽ���Ҷ�
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
//������䷨
Mat countRegion(Mat bw) {
	Mat mask(bw.size(), CV_16U);
	//�޸�Ϊ0-1��ֵͼ��
	bw.convertTo(mask, CV_16U);
	mask /= 255;
	//��ǩ��2��ʼ���㣬��Ϊ���ܻ��ж���256����ͨ����
	int label = 2;
	for (int x = 1; x < bw.rows - 1; x++) {
		for (int y = 1; y < bw.cols - 1; y++) {
			if (mask.at<uint16_t>(x, y) == 1) {
				//�����������ջ
				stack<pair<int, int>> neighborPixels;
				//����ǰ����ֵΪ255���������ջ��
				neighborPixels.push(pair<int, int>(x, y));
				label++;
				//��ջ��Ϊ��ʱ������������ͨ�������ش���
				while(!neighborPixels.empty()) {
					//��õ�ǰ��������
					pair<int, int> nowPixel = neighborPixels.top();
					int now_x = nowPixel.first;
					int now_y = nowPixel.second;
					//��ǰ�����ջ
					neighborPixels.pop();
					//�����ǩ
					mask.at<uint16_t>(now_x, now_y) = label;
					//��������ջ�ж�
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
	//��ʾ��ʹ�ý��������������Ϊ��ǩ�Ǵ�2��ʼ�ģ�����ֵΪ2���������
	imshow("count", mask*5000);

	imshow("origin pic", img);
	imshow("bw pic", bw);
	waitKey();
}