#include <iostream>
#include <vector>
#include <fstream>
using namespace std; 
#include <boost/timer.hpp>

// for sophus 
#include <sophus/se3.hpp>
//using Sophus::SE3d;
using namespace Sophus;

// for eigen 
#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace Eigen;

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

// depth filter with polar line search

// ------------------------------------------------------------------
// parameters 
const int boarder = 20; 	// boarder length
const int width = 640;  	// width
const int height = 480;  	// height

// intrinstic parameter for test_data 
/*const double fx = 481.2f;	
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f; */

// intrinstic parameter for rgb dataset
const double fx = 525.0f;  // focal length x
const double fy = 525.0f;  // focal length y
const double cx = 319.5f;  // optical center x
const double cy = 239.5f;  // optical center y

const int ncc_window_size = 7;	// NCC half window size
const int ncc_area = (2*ncc_window_size+1)*(2*ncc_window_size+1); // NCC window area
const double min_cov = 0.1;	// convergence determination: minimum variance 
const double max_cov = 10;	// divergence determination: maximum variance

// ------------------------------------------------------------------
// functions
// read data from datasets  
bool readDatasetFiles( 
    const string& path, 
    vector<string>& color_image_files, 
    vector<SE3d>& poses 
);

// update the depth according the new image
bool update( 
    const Mat& ref, 
    const Mat& curr, 
    const SE3d& T_C_R, 
    Mat& depth, 
    Mat& depth_cov,
    int x,
    int y,
    bool show 
);

// polar line search
bool epipolarSearch( 
    const Mat& ref, 
    const Mat& curr, 
    const SE3d& T_C_R, 
    const Vector2d& pt_ref, 
    const double& depth_mu, 
    const double& depth_cov,
    Vector2d& pt_curr,
    bool show
);

// update the depth filter
bool updateDepthFilter( 
    const Vector2d& pt_ref, 
    const Vector2d& pt_curr, 
    const SE3d& T_C_R, 
    Mat& depth, 
    Mat& depth_cov
);

// calculate ncc grading
double NCC( const Mat& ref, const Mat& curr, const Vector2d& pt_ref, const Vector2d& pt_curr );

// bilinear grayscale interpolation 
inline double getBilinearInterpolatedValue( const Mat& img, const Vector2d& pt ) {
    uchar* d = & img.data[ int(pt(1,0))*img.step+int(pt(0,0)) ];
    double xx = pt(0,0) - floor(pt(0,0)); 
    double yy = pt(1,0) - floor(pt(1,0));
    return  (( 1-xx ) * ( 1-yy ) * double(d[0]) +
            xx* ( 1-yy ) * double(d[1]) +
            ( 1-xx ) *yy* double(d[img.step]) +
            xx*yy*double(d[img.step+1]))/255.0;
}

// ------------------------------------------------------------------
// visualization functions 
// show depth map
bool plotDepth( const Mat& depth );

// pixel to camera coordinate
inline Vector3d px2cam ( const Vector2d px ) {
    return Vector3d ( 
        (px(0,0) - cx)/fx,
        (px(1,0) - cy)/fy, 
        1
    );
}

// camera coordinate to pixel 
inline Vector2d cam2px ( const Vector3d p_cam ) {
    return Vector2d (
        p_cam(0,0)*fx/p_cam(2,0) + cx, 
        p_cam(1,0)*fy/p_cam(2,0) + cy 
    );
}

// check if a point is in image boundary
inline bool inside( const Vector2d& pt ) {
    return pt(0,0) >= boarder && pt(1,0)>=boarder 
        && pt(0,0)+boarder<width && pt(1,0)+boarder<=height;
}

// show the matched point 
void showEpipolarMatch( const Mat& ref, const Mat& curr, const Vector2d& px_ref, const Vector2d& px_curr );

// show the epipolar line
void showEpipolarLine( const Mat& ref, const Mat& curr, const Vector2d& px_ref, const Vector2d& px_min_curr, const Vector2d& px_max_curr );


// ------------------------------------------------------------------
// main program

int main( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"Usage: dense_mapping path_to_test_dataset"<<endl;
        return -1;
    }
    
    // read images and poses from dataset
    vector<string> color_image_files; 
    vector<SE3d> poses_TWC;
    bool ret = readDatasetFiles( argv[1], color_image_files, poses_TWC );
    if ( ret==false )
    {
        cout<<"Reading image files failed!"<<endl;
        return -1; 
    }
    cout<<"read total "<<color_image_files.size()<<" files."<<endl;
    
    // First image
    Mat ref = imread( color_image_files[0], 0 );                // gray-scale image 
    SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth   = 3.0;    // depth initial value
    double init_cov2    = 5.0;    // covariance initial value 
    Mat depth( height, width, CV_64F, init_depth );             // depth figure
    Mat depth_cov( height, width, CV_64F, init_cov2 );          // depth cov figure

    // initial GFTT points 
    cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(100, 0.01, 10);
    std::vector<cv::KeyPoint> kp;
    detector->detect(ref, kp);

    // plot the search process for each point 
    for( std::vector<cv::KeyPoint>::iterator it = kp.begin(); it != kp.end(); ++it){
    // update for each image
      for ( int index=1; index<color_image_files.size(); index++ )
      {
          cout<<"*** loop "<<index<<" ***"<<endl;
          Mat curr = imread( color_image_files[index], 0 );        
          if (curr.data == nullptr) continue;
          SE3d pose_curr_TWC = poses_TWC[index];
          SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC; // change world coordinate： T_C_W * T_W_R = T_C_R 
          int x = (*it).pt.x;
	  int y = (*it).pt.y;
          // set last parameter to one to show polar line search process 
          update( ref, curr, pose_T_C_R, depth, depth_cov, x, y, false );
          // plotDepth( depth );
          // imshow("image", curr);
          // waitKey(0);
      }
    }

    // write depth value to image
    Mat ref_show;
    cv::cvtColor( ref, ref_show, CV_GRAY2BGR );
    for( std::vector<cv::KeyPoint>::iterator it = kp.begin(); it != kp.end(); ++it){
	int x = (*it).pt.x;
	int y = (*it).pt.y;
        string first_three;
        if( depth_cov.ptr<double>(y)[x] <min_cov ){
          string value = std::to_string((float)depth.ptr<double>(y)[x]);
          first_three = value.substr(0,3);
        }else if( depth_cov.ptr<double>(y)[x] >max_cov ){
	  first_three = "div";
        }else {
          first_three = ""; }

        // string value = std::to_string((float)depth.ptr<double>(y)[x]);
        // string first_three = value.substr(0,3);
        // std::cout<<  std::to_string(depth.ptr<double>(y)[x]) <<std::endl;
        cv::putText(ref_show,
                    first_three,
                    cv::Point(x,y),
		    cv::FONT_HERSHEY_DUPLEX,
		    0.4,
	            CV_RGB(255, 0, 0), //font color
                    0.5
                   );
	cv::circle( ref_show, cv::Point2f(x,y), 2, cv::Scalar(0,255,0), 0.5);
    }  
    cv::imshow("depth value", ref_show);
    cv:waitKey(); 

    //cout<<"estimation returns, saving depth map ..."<<endl;
    //imwrite( "depth.png", depth );
    cout<<"done."<<endl;
    
    return 0;
}


// ------------------------------------------------------------------
// realization of the functions

bool readDatasetFiles(
    const string& path, 
    vector< string >& color_image_files, 
    std::vector<SE3d>& poses
)
{
    ifstream fin( path+"/poses.txt");
    if ( !fin ) return false;
    std::cout << " loading poses successfully " << std::endl;

   
    while ( !fin.eof() )
    {
	// data format：image_name, tx, ty, tz, qx, qy, qz, qw 
        string image; 
        fin>>image;
        std::cout << " loading the image " << image << std::endl; 
        double data[7];
        for ( double& d:data ) fin>>d;
        
        color_image_files.push_back( path+string("/images/")+image );
        poses.push_back(
            SE3d( Quaterniond(data[6], data[3], data[4], data[5]), 
                 Vector3d(data[0], data[1], data[2]))
        );
        if ( !fin.good() ) break;
    }    
    return true;
}

// write depth information
// update depth mat
bool update(const Mat& ref, const Mat& curr, const SE3d& T_C_R, Mat& depth, Mat& depth_cov, int x, int y, bool show )
{
// uncommend to iterate for whole mat
/* #pragma omp parallel for
    for ( int x=boarder; x<width-boarder; x++ )
#pragma omp parallel for
        for ( int y=boarder; y<height-boarder; y++ )
        { */
	    
            // int x = 250; int y = 250;
	    // judge the covariance of the point
            if ( depth_cov.ptr<double>(y)[x] < min_cov ){
               cout << " depth converged " << endl; 
	       return true;	
	    };
            if ( depth_cov.ptr<double>(y)[x] > max_cov ){ 
               cout << " depth  divergered " << endl; 
               return false;
            }
            // continue;
            // search the match in polar line
            Vector2d pt_curr; 
            bool ret = epipolarSearch ( 
                ref, 
                curr, 
                T_C_R, 
                Vector2d(x,y), 
                depth.ptr<double>(y)[x], 
                sqrt(depth_cov.ptr<double>(y)[x]),
                pt_curr,
                show
            );
            
            if ( ret == false ){ // matching fails
               cout << " matching fails " << endl;
               return false;
	    }
                // continue; 
            
	    // uncommend to show match
            if(show == 1)
               showEpipolarMatch( ref, curr, Vector2d(x,y), pt_curr );
            
            // match successful, update the depth filter
            updateDepthFilter( Vector2d(x,y), pt_curr, T_C_R, depth, depth_cov );
            cout << " after updating depth is " << depth.ptr<double>(y)[x] << endl;
            cout << " after updating convariance is " << depth_cov.ptr<double>(y)[x] << endl;
        // }
}

// polar line search
bool epipolarSearch(
    const Mat& ref, const Mat& curr, 
    const SE3d& T_C_R, const Vector2d& pt_ref, 
    const double& depth_mu, const double& depth_cov, 
    Vector2d& pt_curr, bool show )
{
    Vector3d f_ref = px2cam( pt_ref );
    f_ref.normalize();
    Vector3d P_ref = f_ref*depth_mu;	// the p vecotr in reference frame
    
    Vector2d px_mean_curr = cam2px( T_C_R*P_ref ); // the projected pixel according the depth average
    double d_min = depth_mu-4*depth_cov, d_max = depth_mu+4*depth_cov;
    if ( d_min<0.1 ) d_min = 0.1;
    Vector2d px_min_curr = cam2px( T_C_R*(f_ref*d_min) );	// the projected pixel according the smallest depth value
    Vector2d px_max_curr = cam2px( T_C_R*(f_ref*d_max) );	// the projected pixel according the smallest depth value
    
    Vector2d epipolar_line = px_max_curr - px_min_curr;	// epipolar line 
    Vector2d epipolar_direction = epipolar_line;		// direction of epipolar line
    epipolar_direction.normalize();
    double half_length = 0.5*epipolar_line.norm();	// half length of epipolar line
    if ( half_length>100 ) half_length = 100;   // dont search too much
    
    // show epipolar line
    if(show == 1)
        showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );
    
    // search around the epipolar line
    double best_ncc = -1.0;
    Vector2d best_px_curr; 
    for ( double l=-half_length; l<=half_length; l+=0.7 )  // l+=sqrt(2) 
    {
        Vector2d px_curr = px_mean_curr + l*epipolar_direction;  // points to be matched
        if ( !inside(px_curr) )
            continue; 
        // calculated NCC
        double ncc = NCC( ref, curr, pt_ref, px_curr );
        if ( ncc>best_ncc )
        {
            best_ncc = ncc; 
            best_px_curr = px_curr;
        }
    }
    if ( best_ncc < 0.95f )      // only choose large ncc values
        return false; 
    pt_curr = best_px_curr;
    return true;
}

double NCC (
    const Mat& ref, const Mat& curr, 
    const Vector2d& pt_ref, const Vector2d& pt_curr
)
{
    // average should be zero
    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_curr; // average value of reference frame and current frame
    for ( int x=-ncc_window_size; x<=ncc_window_size; x++ )
        for ( int y=-ncc_window_size; y<=ncc_window_size; y++ )
        {
            double value_ref = double(ref.ptr<uchar>( int(y+pt_ref(1,0)) )[ int(x+pt_ref(0,0)) ])/255.0;
            mean_ref += value_ref;
            
            double value_curr = getBilinearInterpolatedValue( curr, pt_curr+Vector2d(x,y) );
            mean_curr += value_curr;
            
            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }
        
    mean_ref /= ncc_area;
    mean_curr /= ncc_area;
    
	// calculate Zero mean NCC
    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for ( int i=0; i<values_ref.size(); i++ )
    {
        double n = (values_ref[i]-mean_ref) * (values_curr[i]-mean_curr);
        numerator += n;
        demoniator1 += (values_ref[i]-mean_ref)*(values_ref[i]-mean_ref);
        demoniator2 += (values_curr[i]-mean_curr)*(values_curr[i]-mean_curr);
    }
    return numerator / sqrt( demoniator1*demoniator2+1e-10 );   // numerical stability
}

bool updateDepthFilter(
    const Vector2d& pt_ref, 
    const Vector2d& pt_curr, 
    const SE3d& T_C_R,
    Mat& depth, 
    Mat& depth_cov
)
{
    // depth calculation using triangulation
    SE3d T_R_C = T_C_R.inverse();
    Vector3d f_ref = px2cam( pt_ref );
    f_ref.normalize();
    Vector3d f_curr = px2cam( pt_curr );
    f_curr.normalize();
    
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // => [ f_ref^T f_ref, -f_ref^T f_cur ] [d_ref] = [f_ref^T t]
    //    [ f_cur^T f_ref, -f_cur^T f_cur ] [d_cur] = [f_cur^T t]

    Vector3d t = T_R_C.translation();
    Vector3d f2 = T_R_C.rotationMatrix() * f_curr; 
    //Vector3d f2 = T_R_C.rotation() * f_curr; 
    Vector2d b = Vector2d ( t.dot ( f_ref ), t.dot ( f2 ) );
    double A[4];
    A[0] = f_ref.dot ( f_ref );
    A[2] = f_ref.dot ( f2 );
    A[1] = -A[2];
    A[3] = - f2.dot ( f2 );
    double d = A[0]*A[3]-A[1]*A[2];
    Vector2d lambdavec = 
        Vector2d (  A[3] * b ( 0,0 ) - A[1] * b ( 1,0 ),
                    -A[2] * b ( 0,0 ) + A[0] * b ( 1,0 )) /d;
    Vector3d xm = lambdavec ( 0,0 ) * f_ref;
    Vector3d xn = t + lambdavec ( 1,0 ) * f2;
    Vector3d d_esti = ( xm+xn ) / 2.0;  // depth vector calculated by triangulation
    double depth_estimation = d_esti.norm();   // depth value
    
    // uncertainty calculation (one pixel as error)
    Vector3d p = f_ref*depth_estimation;
    Vector3d a = p - t; 
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos( f_ref.dot(t)/t_norm );
    double beta = acos( -a.dot(t)/(a_norm*t_norm));
    double beta_prime = beta + atan(1/fx);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation; 
    double d_cov2 = d_cov*d_cov;
    
    // gaussian fusion
    double mu = depth.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ];
    double sigma2 = depth_cov.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ];
    
    double mu_fuse = (d_cov2*mu+sigma2*depth_estimation) / ( sigma2+d_cov2);
    double sigma_fuse2 = ( sigma2 * d_cov2 ) / ( sigma2 + d_cov2 );
    
    depth.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ] = mu_fuse; 
    depth_cov.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ] = sigma_fuse2;
    
    return true;
}

// visualization functions
bool plotDepth(const Mat& depth)
{
    imshow( "depth", depth*0.4 );
    waitKey(1);
}

void showEpipolarMatch(const Mat& ref, const Mat& curr, const Vector2d& px_ref, const Vector2d& px_curr)
{
    Mat ref_show, curr_show;
    cv::cvtColor( ref, ref_show, CV_GRAY2BGR );
    cv::cvtColor( curr, curr_show, CV_GRAY2BGR );
    
    cv::circle( ref_show, cv::Point2f(px_ref(0,0), px_ref(1,0)), 5, cv::Scalar(0,0,250), 2);
    cv::circle( curr_show, cv::Point2f(px_curr(0,0), px_curr(1,0)), 5, cv::Scalar(0,0,250), 2);
    
    imshow("ref", ref_show );
    imshow("curr", curr_show );
    waitKey(0);
}

void showEpipolarLine(const Mat& ref, const Mat& curr, const Vector2d& px_ref, const Vector2d& px_min_curr, const Vector2d& px_max_curr)
{

    Mat ref_show, curr_show;
    cv::cvtColor( ref, ref_show, CV_GRAY2BGR );
    cv::cvtColor( curr, curr_show, CV_GRAY2BGR );
    
    cv::circle( ref_show, cv::Point2f(px_ref(0,0), px_ref(1,0)), 5, cv::Scalar(0,255,0), 2);
    cv::circle( curr_show, cv::Point2f(px_min_curr(0,0), px_min_curr(1,0)), 5, cv::Scalar(0,255,0), 2);
    cv::circle( curr_show, cv::Point2f(px_max_curr(0,0), px_max_curr(1,0)), 5, cv::Scalar(0,255,0), 2);
    cv::line( curr_show, Point2f(px_min_curr(0,0), px_min_curr(1,0)), Point2f(px_max_curr(0,0), px_max_curr(1,0)), Scalar(0,255,0), 1);
    
    imshow("ref", ref_show );
    imshow("curr", curr_show );
    waitKey(0);
}

