/**
 * File: TFSolver.h
 * Project: DVision library
 * Author: Anurag Vempati
 * Date: Match 11, 2017
 * Description: Computes TF (Rotation, Translation) matrices
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_TF_SOLVER__
#define __D_TF_SOLVER__

#include <opencv2/core/core.hpp>
#include <vector>

#define EPSILON 0.0001

namespace DVision {

/// Computes TF matrices
class TFSolver
{
public:

  /**
   * Creates the solver
   */
  TFSolver();
  
  /**
   * Destructor
   */
  virtual ~TFSolver(){}
  
  /**
   * Finds a TF matrix from the given Fundamental matrix
   * @param pts1,2 matched keypoints from image 1 and 2 respectively
   * @param pts2_depth depth of matched keypoints in image 2
   * @param K intrinics of the camera
   * @param T1 pose of camera 1
   * @param F Fundamental matrix
   * @return T2 pose of camera 2
   */
  cv::Mat findTFMat(
      const cv::Mat &pts1, const cv::Mat &pts2, const cv::Mat &pts2_depth,
      const cv::Mat &K, const cv::Mat &T1,
      const cv::Mat &F);

protected:

  /**
   * SVD decomposition of Essential Matrix
   * @param E Essential Matrix
   * @return svd_* SVD components
   */
  void TakeSVDOfE(cv::Mat_<double>& E, cv::Mat& svd_u, cv::Mat& svd_vt, cv::Mat& svd_w);

  /**
   * Evaluate 4 possible pairs of [R, t] from Essential Matrix E
   * @param E Essential Matrix
   * @return R1,R2,t1,t2 4 2 possible R and t each for the candidate TF
   */
  bool DecomposeEtoRandT(
      cv::Mat_<double>& E, cv::Mat_<double>& R1, cv::Mat_<double>& R2,
      cv::Mat_<double>& t1, cv::Mat_<double>& t2);

  /**
   * Sanity check for rotation matrix
   * @param R rotation matrix
   * @return whether or not coherent
   */
  bool CheckCoherentRotation(cv::Mat_<double>& R);

  cv::Mat_<double> LinearLSTriangulation(
      cv::Point3d u,   //homogenous image point (u,v,1)
      cv::Matx44d P,   //camera 1 matrix
      cv::Point3d u1,    //homogenous image point in 2nd camera
      cv::Matx44d P1);

  cv::Mat_<double> IterativeLinearLSTriangulation(
      cv::Point3d u,  //homogenous image point (u,v,1)
      cv::Matx44d P,      //camera 1 matrix
      cv::Point3d u1,     //homogenous image point in 2nd camera
      cv::Matx44d P1);

  //Triagulate points
  // returns reprojection error and scale of the scene
  std::pair<double, double> TriangulatePoints(
      const cv::Mat& pt_set1, const cv::Mat& pt_set2,
      const cv::Mat &pts1_depth, const cv::Mat &pts2_depth,
      const cv::Mat& K, const cv::Mat& Kinv, const cv::Mat& T1,
      const cv::Mat& T2, cv::Mat& pointcloud);

  /**
   * Test if the 3D points triangulate infront of the camera
   * @param pcloud point cloud 4xN
   * @param T pose of camera
   * @return status bool indicating if that point triangulates infront
   */
  bool TestTriangulation(
      const cv::Mat_<double>& pcloud, const cv::Mat_<double>& T, std::vector<uchar>& status);

};

} // namespace DVision

#endif
