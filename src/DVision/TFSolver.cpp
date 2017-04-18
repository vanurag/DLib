/**
 * File: TFSolver.cpp
 * Project: DVision library
 * Author: Dorian Galvez-Lopez
 * Date: November 17, 2011
 * Description: Computes fundamental matrices
 * License: see the LICENSE.txt file
 *
 */
 
#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include "TFSolver.h"

#include "DUtils.h"
#include "DUtilsCV.h"

using namespace std;
using namespace DUtils;
using namespace DUtilsCV;

namespace DVision {

// ----------------------------------------------------------------------------

TFSolver::TFSolver()
{
}

// ----------------------------------------------------------------------------

void TFSolver::TakeSVDOfE(
    cv::Mat_<double>& E, cv::Mat& svd_u, cv::Mat& svd_vt, cv::Mat& svd_w)
{
  //Using OpenCV's SVD
  cv::SVD svd(E, cv::SVD::MODIFY_A);
  svd_u = svd.u;
  svd_vt = svd.vt;
  svd_w = svd.w;

  std::cout << "----------------------- SVD ------------------------\n";
  std::cout << "U:\n"<<svd_u<<"\nW:\n"<<svd_w<<"\nVt:\n"<<svd_vt<<std::endl;
  std::cout << "----------------------------------------------------\n";
}

// --------------------------------------------------------------------------

bool TFSolver::DecomposeEtoRandT(
  cv::Mat_<double>& E, cv::Mat_<double>& R1, cv::Mat_<double>& R2,
  cv::Mat_<double>& t1, cv::Mat_<double>& t2)
{
  //Using HZ E decomposition
  cv::Mat svd_u, svd_vt, svd_w;
  TakeSVDOfE(E,svd_u,svd_vt,svd_w);

  //check if first and second singular values are the same (as they should be)
  double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
  if(singular_values_ratio>1.0) singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
  if (singular_values_ratio < 0.7) {
    cout << "singular values are too far apart\n";
    return false;
  }

  cv::Matx33d W(0,-1,0, //HZ 9.13
    1,0,0,
    0,0,1);
  cv::Matx33d Wt(0,1,0,
    -1,0,0,
    0,0,1);
  R1 = svd_u * cv::Mat(W) * svd_vt; //HZ 9.19
  R2 = svd_u * cv::Mat(Wt) * svd_vt; //HZ 9.19
  t1 = svd_u.col(2); //u3
  t2 = -svd_u.col(2); //u3

  return true;
}

// --------------------------------------------------------------------------

bool TFSolver::CheckCoherentRotation(cv::Mat_<double>& R) {


  if(fabsf(cv::determinant(R))-1.0 > 1e-07) {
    std::cerr << "det(R) != +-1.0 [" << fabsf(cv::determinant(R)) << "], this is not a rotation matrix" << std::endl;
    return false;
  }

  return true;
}

//

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> TFSolver::LinearLSTriangulation(
    cv::Point3d u,   //homogenous image point (u,v,1)
    cv::Matx44d P,   //camera 1 matrix
    cv::Point3d u1,    //homogenous image point in 2nd camera
    cv::Matx44d P1)   //camera 2 matrix
{

  //build matrix A for homogenous equation system Ax = 0
  //assume X = (x,y,z,1), for Linear-LS method
  //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
  //  cout << "u " << u <<", u1 " << u1 << endl;
  //  Matx<double,6,4> A; //this is for the AX=0 case, and with linear dependence..
  //  A(0) = u.x*P(2)-P(0);
  //  A(1) = u.y*P(2)-P(1);
  //  A(2) = u.x*P(1)-u.y*P(0);
  //  A(3) = u1.x*P1(2)-P1(0);
  //  A(4) = u1.y*P1(2)-P1(1);
  //  A(5) = u1.x*P(1)-u1.y*P1(0);
  //  Matx43d A; //not working for some reason...
  //  A(0) = u.x*P(2)-P(0);
  //  A(1) = u.y*P(2)-P(1);
  //  A(2) = u1.x*P1(2)-P1(0);
  //  A(3) = u1.y*P1(2)-P1(1);
  cv::Matx43d A(u.x*P(2,0)-P(0,0),  u.x*P(2,1)-P(0,1),    u.x*P(2,2)-P(0,2),
        u.y*P(2,0)-P(1,0),  u.y*P(2,1)-P(1,1),    u.y*P(2,2)-P(1,2),
        u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1), u1.x*P1(2,2)-P1(0,2),
        u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1), u1.y*P1(2,2)-P1(1,2)
        );
  cv::Matx41d B(-(u.x*P(2,3)  -P(0,3)),
        -(u.y*P(2,3)  -P(1,3)),
        -(u1.x*P1(2,3)  -P1(0,3)),
        -(u1.y*P1(2,3)  -P1(1,3)));

  cv::Mat_<double> X;
  cv::solve(A,B,X,cv::DECOMP_SVD);

  return X;
}

// ----------------------------------------------------------------------------

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> TFSolver::IterativeLinearLSTriangulation(
    cv::Point3d u,  //homogenous image point (u,v,1)
    cv::Matx44d P,      //camera 1 matrix
    cv::Point3d u1,     //homogenous image point in 2nd camera
    cv::Matx44d P1)      //camera 2 matrix
{
  double wi = 1, wi1 = 1;
  cv::Mat_<double> X(4,1);

  cv::Mat_<double> X_ = LinearLSTriangulation(u,P,u1,P1);
  X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;

  cv::Mat_<double> B(4,1);
  for (int i=0; i<10; i++) { //Hartley suggests 10 iterations at most

    //recalculate weights
    double p2x = cv::Mat_<double>(cv::Mat_<double>(P).row(2)*X)(0);
    double p2x1 = cv::Mat_<double>(cv::Mat_<double>(P1).row(2)*X)(0);

    //breaking point
    if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

    wi = p2x;
    wi1 = p2x1;

    //reweight equations and solve
    cv::Matx43d A((u.x*P(2,0)-P(0,0))/wi,   (u.x*P(2,1)-P(0,1))/wi,     (u.x*P(2,2)-P(0,2))/wi,
          (u.y*P(2,0)-P(1,0))/wi,   (u.y*P(2,1)-P(1,1))/wi,     (u.y*P(2,2)-P(1,2))/wi,
          (u1.x*P1(2,0)-P1(0,0))/wi1, (u1.x*P1(2,1)-P1(0,1))/wi1,   (u1.x*P1(2,2)-P1(0,2))/wi1,
          (u1.y*P1(2,0)-P1(1,0))/wi1, (u1.y*P1(2,1)-P1(1,1))/wi1,   (u1.y*P1(2,2)-P1(1,2))/wi1
          );
    B.at<double>(0,0) = -(u.x*P(2,3)  -P(0,3))/wi;
    B.at<double>(1,0) = -(u.y*P(2,3)  -P(1,3))/wi;
    B.at<double>(2,0) = -(u1.x*P1(2,3)  -P1(0,3))/wi1;
    B.at<double>(3,0) = -(u1.y*P1(2,3)  -P1(1,3))/wi1;

    cv::solve(A,B,X_,cv::DECOMP_SVD);
    X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
  }
  return X;
}

// ----------------------------------------------------------------------------

//Triagulate points
std::pair<double, double> TFSolver::TriangulatePoints(
    const cv::Mat& pt_set1, const cv::Mat& pt_set2,
    const cv::Mat &pts1_depth, const cv::Mat &pts2_depth,
    const cv::Mat& K, const cv::Mat& Kinv, const cv::Mat& T1,
    const cv::Mat& T2, cv::Mat& pointcloud)
{
  std::cout << "Triangulating..." << std::endl;
  std::vector<double> reproj_error;
  double scale = 0.0;
  int num_scale = 0; // num of estimates involved in evaluating scale
  unsigned int pts_size = pt_set1.rows;

  std::cout << "K, T1: " << K << "\n" << T1 << std::endl;
  std::cout << "K, T2: " << K << "\n" << T2 << std::endl;
  cv::Mat_<double> KT1 = K * T1(cv::Rect(0,0,4,3));
  cv::Mat_<double> KT2 = K * T2(cv::Rect(0,0,4,3));
//#pragma omp parallel for num_threads(1)
  for (int i=0; i<pts_size; i++) {
    cv::Mat p1 = pt_set1.row(i);
    cv::Point3d u1(p1.at<double>(0),p1.at<double>(1),1.0);
    cv::Mat_<double> um1 = Kinv * cv::Mat_<double>(u1);
    u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);

    cv::Mat p2 = pt_set2.row(i);
    cv::Point3d u2(p2.at<double>(0),p2.at<double>(1),1.0);
    cv::Mat_<double> um2 = Kinv * cv::Mat_<double>(u2);
    u2.x = um2(0); u2.y = um2(1); u2.z = um2(2);

    cv::Mat_<double> X = IterativeLinearLSTriangulation(u1, T1, u2, T2);

    // estimate scale
    cv::Mat_<double> xPt_i;
    if (!pts1_depth.empty() && pts1_depth.at<double>(i) > 0) {
      xPt_i = T1(cv::Rect(0,0,4,3)) * X;
      scale = ( (num_scale * scale) + (xPt_i(2) / pts1_depth.at<double>(i)) ) / (num_scale + 1);
    } else if (!pts2_depth.empty() && pts2_depth.at<double>(i) > 0) {
      xPt_i = T2(cv::Rect(0,0,4,3)) * X;
      scale = ( (num_scale * scale) + (xPt_i(2) / pts2_depth.at<double>(i)) ) / (num_scale + 1);
    }
    num_scale += 1;
    std::cout << "curr scale estimate: " << scale << std::endl;

    cv::Mat_<double> xPt_img_2 = KT2 * X;       //reproject
    cv::Point2f xPt_img_2_(xPt_img_2(0)/xPt_img_2(2),xPt_img_2(1)/xPt_img_2(2));

//#pragma omp critical
//    {
//    std::cout << "err terms: " << xPt_img_2_ << "\n" << cv::Point2f(p2.at<double>(0),p2.at<double>(1)) << std::endl;
    double reprj_err = cv::norm(xPt_img_2_-cv::Point2f(p2.at<double>(0),p2.at<double>(1)));
    std::cout << "rp error: " << reprj_err << std::endl;
    reproj_error.push_back(reprj_err);

    pointcloud.at<double>(0,i) = X(0);
    pointcloud.at<double>(1,i) = X(1);
    pointcloud.at<double>(2,i) = X(2);
    pointcloud.at<double>(3,i) = 1.0;
//    }
  }
  std::cout << "final scale estimate [no. estimates]: " << scale << " [" << num_scale << "]" << std::endl;

  cv::Scalar mse = cv::mean(reproj_error);
  std::cout << "Done. ("<<pointcloud.size()<<"points, mean reproj err = " << mse[0] << ")"<< std::endl;

  //show "range image"
#ifdef __SFM__DEBUG__
  {
    double minVal,maxVal;
    minMaxLoc(depths, &minVal, &maxVal);
    Mat tmp(240,320,CV_8UC3,Scalar(0,0,0)); //cvtColor(img_1_orig, tmp, CV_BGR2HSV);
    for (unsigned int i=0; i<pointcloud.size(); i++) {
      double _d = MAX(MIN((pointcloud[i].z-minVal)/(maxVal-minVal),1.0),0.0);
      circle(tmp, correspImg1Pt[i].pt, 1, Scalar(255 * (1.0-(_d)),255,255), CV_FILLED);
    }
    cvtColor(tmp, tmp, CV_HSV2BGR);
    imshow("Depth Map", tmp);
    waitKey(0);
    destroyWindow("Depth Map");
  }
#endif

  return std::make_pair(mse[0], scale);
}

// ----------------------------------------------------------------------------

bool TFSolver::TestTriangulation(
    const cv::Mat_<double>& pcloud, const cv::Mat_<double>& T, std::vector<uchar>& status)
{
  cv::Mat_<double> p_cloud_projected = T * pcloud;
  std::cout << "p_cloud_proj: " << p_cloud_projected(cv::Rect(0,2,pcloud.cols,1)) << std::endl;

  int count = 0;
  status.resize(pcloud.cols,0);
  for (int i=0; i<pcloud.cols; i++) {
    status[i] = (p_cloud_projected.at<double>(2, i) > 0) ? 1 : 0;
    if (status[i] != 0) count += 1;
  }

  double percentage = ((double)count / (double)pcloud.cols);
  std::cout << count << "/" << pcloud.cols << " = " << percentage*100.0 << "% are in front of camera" << std::endl;
  if(percentage < 0.8)
    return false; //less than 80% of the points are in front of the camera

  return true;
}

// --------------------------------------------------------------------------

cv::Mat TFSolver::findTFMat(
    const cv::Mat &pts1, const cv::Mat &pts2, const cv::Mat &pts2_depth,
    const cv::Mat &K, const cv::Mat &T1, const cv::Mat &F)
{
  cv::Mat TF = cv::Mat::zeros(4, 4, CV_64F);
  if (pts1.rows != pts2.rows) return TF;
  std::cout << "PR F: " << F << std::endl;
  std::cout << "PR K: " << K << std::endl;
  cv::Mat_<double> E = K.t() * F * K;

  if(fabsf(cv::determinant(E)) > 1e-07) {
    std::cout << "det(E) != 0 : " << cv::determinant(E) << "\n";
    return TF;
  }

  cv::Mat_<double> R1(3,3);
  cv::Mat_<double> R2(3,3);
  cv::Mat_<double> t1(1,3);
  cv::Mat_<double> t2(1,3);

  if (!DecomposeEtoRandT(E,R1,R2,t1,t2)) return TF;

  if(cv::determinant(R1)+1.0 < 1e-09) {
    //according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
    std::cout << "det(R) == -1 ["<<cv::determinant(R1)<<"]: flip E's sign" << std::endl;
    E = -E;
    DecomposeEtoRandT(E,R1,R2,t1,t2);
  }
  if (!CheckCoherentRotation(R1)) {
    std::cout << "resulting rotation R1 is not coherent\n";
    return TF;
  }

  std::cout << "R1: \n" << R1 << std::endl;
  std::cout << "R2: \n" << R2 << std::endl;

  // Case #1
  TF = (cv::Mat_<double>(4, 4) << // TF from 2 to 1
      R1(0,0), R1(0,1),  R1(0,2),  t1(0),
      R1(1,0), R1(1,1),  R1(1,2),  t1(1),
      R1(2,0), R1(2,1),  R1(2,2),  t1(2),
            0,       0,        0,      1);
  std::cout << std::endl << "Testing Case #1 " << std::endl << cv::Mat(TF) << std::endl;

  cv::Mat_<double> pcloud1(4, pts1.rows);
  cv::Mat_<double> pcloud2(4, pts2.rows);
  std::pair<double, double> tri_status1 = TriangulatePoints(pts1, pts2, cv::Mat(), pts2_depth, K, K.inv(), TF, cv::Mat::eye(4,4, CV_64F), pcloud1);
  std::pair<double, double> tri_status2 = TriangulatePoints(pts2, pts1, pts2_depth, cv::Mat(), K, K.inv(), cv::Mat::eye(4,4, CV_64F), TF, pcloud2);
  std::vector<uchar> tmp_status;

  std::cout << "pcl1: \n" << pcloud1 << std::endl;
  std::cout << "pcl2: \n" << pcloud2 << std::endl;
  TestTriangulation(pcloud1, TF, tmp_status);
  TestTriangulation(pcloud2, cv::Mat::eye(4,4, CV_64F), tmp_status);

  //check if points are triangulated --in front-- of cameras for all 4 ambiguations
  if (!TestTriangulation(pcloud1, TF, tmp_status) || !TestTriangulation(pcloud2, cv::Mat::eye(4,4, CV_64F), tmp_status) || tri_status1.first > 2.0 || tri_status2.first > 2.0) {
    // Case #2
    TF = (cv::Mat_<double>(4, 4) <<
        R1(0,0), R1(0,1),  R1(0,2),  t2(0),
        R1(1,0), R1(1,1),  R1(1,2),  t2(1),
        R1(2,0), R1(2,1),  R1(2,2),  t2(2),
              0,       0,        0,      1);
    std::cout << std::endl << "Testing Case #2 "<< std::endl << cv::Mat(TF) << std::endl;

    tri_status1 = TriangulatePoints(pts1, pts2, cv::Mat(), pts2_depth, K, K.inv(), TF, cv::Mat::eye(4,4, CV_64F), pcloud1);
    tri_status2 = TriangulatePoints(pts2, pts1, pts2_depth, cv::Mat(), K, K.inv(), cv::Mat::eye(4,4, CV_64F), TF, pcloud2);

    std::cout << "pcl1: \n" << pcloud1 << std::endl;
    std::cout << "pcl2: \n" << pcloud2 << std::endl;
    TestTriangulation(pcloud1, TF, tmp_status);
    TestTriangulation(pcloud2, cv::Mat::eye(4,4, CV_64F), tmp_status);

    if (!TestTriangulation(pcloud1, TF, tmp_status) || !TestTriangulation(pcloud2, cv::Mat::eye(4,4, CV_64F), tmp_status) || tri_status1.first > 2.0 || tri_status2.first > 2.0) {
      if (!CheckCoherentRotation(R2)) {
        std::cout << "resulting rotation R2 is not coherent\n";
        TF.setTo(0);
        return TF;
      }
      // Case #3
      TF = (cv::Mat_<double>(4, 4) <<
          R2(0,0), R2(0,1),  R2(0,2),  t1(0),
          R2(1,0), R2(1,1),  R2(1,2),  t1(1),
          R2(2,0), R2(2,1),  R2(2,2),  t1(2),
                0,       0,        0,      1);
      std::cout << std::endl << "Testing Case #3 "<< std::endl << cv::Mat(TF) << std::endl;

      tri_status1 = TriangulatePoints(pts1, pts2, cv::Mat(), pts2_depth, K, K.inv(), TF, cv::Mat::eye(4,4, CV_64F), pcloud1);
      tri_status2 = TriangulatePoints(pts2, pts1, pts2_depth, cv::Mat(), K, K.inv(), cv::Mat::eye(4,4, CV_64F), TF, pcloud2);

      std::cout << "pcl1: \n" << pcloud1 << std::endl;
      std::cout << "pcl2: \n" << pcloud2 << std::endl;
      TestTriangulation(pcloud1, TF, tmp_status);
      TestTriangulation(pcloud2, cv::Mat::eye(4,4, CV_64F), tmp_status);

      if (!TestTriangulation(pcloud1, TF, tmp_status) || !TestTriangulation(pcloud2, cv::Mat::eye(4,4, CV_64F), tmp_status) || tri_status1.first > 2.0 || tri_status2.first > 2.0) {
        // Case #4
        TF = (cv::Mat_<double>(4, 4) <<
            R2(0,0), R2(0,1),  R2(0,2),  t2(0),
            R2(1,0), R2(1,1),  R2(1,2),  t2(1),
            R2(2,0), R2(2,1),  R2(2,2),  t2(2),
                  0,       0,        0,      1);
        std::cout << std::endl << "Testing Case #4 "<< std::endl << cv::Mat(TF) << std::endl;

        tri_status1 = TriangulatePoints(pts1, pts2, cv::Mat(), pts2_depth, K, K.inv(), TF, cv::Mat::eye(4,4, CV_64F), pcloud1);
        tri_status2 = TriangulatePoints(pts2, pts1, pts2_depth, cv::Mat(), K, K.inv(), cv::Mat::eye(4,4, CV_64F), TF, pcloud2);

        std::cout << "pcl1: \n" << pcloud1 << std::endl;
        std::cout << "pcl2: \n" << pcloud2 << std::endl;
        TestTriangulation(pcloud1, TF, tmp_status);
        TestTriangulation(pcloud2, cv::Mat::eye(4,4, CV_64F), tmp_status);

        if (!TestTriangulation(pcloud1, TF, tmp_status) || !TestTriangulation(pcloud2, cv::Mat::eye(4,4, CV_64F), tmp_status) || tri_status1.first > 2.0 || tri_status2.first > 2.0) {
          std::cout << "None of the configurations worked." << std::endl;
          TF.setTo(0);
          return TF;
        } // Case #4

      } // Case #3
    } // Case #2
  } // Case #1

  std::cout << "find TF mat TF: " << TF << std::endl;
  if (tri_status1.second > 0 && tri_status2.second > 0) {
    // scaling
    double scale = (tri_status1.second + tri_status2.second) / 2.0;
    TF(cv::Rect(3,0,1,3)) /= scale;
    std::cout << "find TF mat TF at scale: " << TF << std::endl;
    std::cout << "find TF mat TF_inv: " << TF.inv() << std::endl;
    std::cout << "find TF mat T1: " << T1 << std::endl;
    if (T1.type() == CV_64F) {
      std::cout << "find TF mat TF_inv * T1: " << TF.inv() * T1 << std::endl;
      return TF.inv() * T1;
    } else {
      cv::Mat_<double> T1_d;
      T1.convertTo(T1_d, CV_64F);
      std::cout << "find TF mat T1_d: " << T1_d << std::endl;
      std::cout << "find TF mat TF_inv * T1_d: " << TF.inv() * T1_d << std::endl;
      return TF.inv() * T1_d;
    }
  } else {
    TF.setTo(0);
    return TF;
  }
}

// --------------------------------------------------------------------------



 } // namespace TFSolver
