package ai.magicmirror.magicmirror_faceswap_nodlib.HairSwap;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Point;

import org.apache.commons.math3.linear.MatrixUtils;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opencv.core.Core.BORDER_TRANSPARENT;
import static org.opencv.core.Core.FILLED;
import static org.opencv.core.Core.SVDecomp;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_32FC2;
import static org.opencv.core.CvType.CV_64F;
import static org.opencv.core.CvType.CV_8S;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.CvType.CV_8UC;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.core.CvType.CV_8UC4;
import static org.opencv.imgproc.Imgproc.WARP_INVERSE_MAP;
import static org.opencv.imgproc.Imgproc.boundingRect;
import static org.opencv.imgproc.Imgproc.convexHull;
import static org.opencv.imgproc.Imgproc.drawContours;
import static org.opencv.imgproc.Imgproc.fillConvexPoly;
import static org.opencv.imgproc.Imgproc.fillPoly;
import static org.opencv.imgproc.Imgproc.findContours;
import static org.opencv.imgproc.Imgproc.floodFill;
import static org.opencv.imgproc.Imgproc.line;
import static org.opencv.imgproc.Imgproc.polylines;
import static org.opencv.imgproc.Imgproc.warpAffine;
import static org.opencv.photo.Photo.NORMAL_CLONE;

/**
 * Created by seven on 3/16/18.
 */

public class FaceSwap_mat {

    private static final String TAG = "FACE SWAP";

//    private FaceDet mFaceDet = null;

    private static FaceSwap_mat faceSwap = null;
    private Mat hairModelImageMat;
    private Mat selfieImageMat;
    private Context context;

    private FaceSwap_mat(Context context) {
        this.context = context;
    }

    public static FaceSwap_mat getInstance(Context context){
        if(null == faceSwap){
            faceSwap = new FaceSwap_mat(context);
        }

        return faceSwap;
    }

    public Mat swap(Mat selfieImage, Mat hairModelImage) {
        System.out.println("swap called");

//        hairModelImageMat = new Mat(hairModelImage.getHeight(), hairModelImage.getWidth(), CV_8UC3);
//        selfieImageMat = new Mat(selfieImage.getHeight(), selfieImage.getWidth(), CV_8UC(3));
//
//        Utils.bitmapToMat(selfieImage, selfieImageMat);
//        Utils.bitmapToMat(hairModelImage, hairModelImageMat);
//
//        System.out.println(hairModelImageMat.channels() + "");
//        System.out.println(selfieImageMat.channels() + "");

        selfieImageMat = selfieImage;
        hairModelImageMat = hairModelImage;

        ArrayList<Point> selfieImageLandmarks = new ArrayList<>();
        ArrayList<Point> hairModelImageLandmark = new ArrayList<>();

        selfieImageLandmarks.add(new Point(38, 146));
        selfieImageLandmarks.add(new Point(39, 167));
        selfieImageLandmarks.add(new Point(42, 188));
        selfieImageLandmarks.add(new Point(46, 208));
        selfieImageLandmarks.add(new Point(53, 227));
        selfieImageLandmarks.add(new Point(64, 243));
        selfieImageLandmarks.add(new Point(80, 256));
        selfieImageLandmarks.add(new Point(99, 265));
        selfieImageLandmarks.add(new Point(120, 268));
        selfieImageLandmarks.add(new Point(141, 266));
        selfieImageLandmarks.add(new Point(160, 256));
        selfieImageLandmarks.add(new Point(177, 244));
        selfieImageLandmarks.add(new Point(189, 228));
        selfieImageLandmarks.add(new Point(197, 208));
        selfieImageLandmarks.add(new Point(200, 187));
        selfieImageLandmarks.add(new Point(203, 165));
        selfieImageLandmarks.add(new Point(204, 144));
        selfieImageLandmarks.add(new Point(53, 127));
        selfieImageLandmarks.add(new Point(63, 119));
        selfieImageLandmarks.add(new Point(77, 118));
        selfieImageLandmarks.add(new Point(90, 120));
        selfieImageLandmarks.add(new Point(104, 125));
        selfieImageLandmarks.add(new Point(132, 125));
        selfieImageLandmarks.add(new Point(146, 120));
        selfieImageLandmarks.add(new Point(160, 117));
        selfieImageLandmarks.add(new Point(175, 118));
        selfieImageLandmarks.add(new Point(187, 125));
        selfieImageLandmarks.add(new Point(118, 140));
        selfieImageLandmarks.add(new Point(118, 153));
        selfieImageLandmarks.add(new Point(118, 165));
        selfieImageLandmarks.add(new Point(118, 178));
        selfieImageLandmarks.add(new Point(106, 190));
        selfieImageLandmarks.add(new Point(112, 192));
        selfieImageLandmarks.add(new Point(119, 193));
        selfieImageLandmarks.add(new Point(126, 192));
        selfieImageLandmarks.add(new Point(133, 190));
        selfieImageLandmarks.add(new Point(68, 143));
        selfieImageLandmarks.add(new Point(77, 136));
        selfieImageLandmarks.add(new Point(89, 136));
        selfieImageLandmarks.add(new Point(99, 144));
        selfieImageLandmarks.add(new Point(88, 148));
        selfieImageLandmarks.add(new Point(77, 148));
        selfieImageLandmarks.add(new Point(140, 144));
        selfieImageLandmarks.add(new Point(150, 136));
        selfieImageLandmarks.add(new Point(162, 136));
        selfieImageLandmarks.add(new Point(172, 143));
        selfieImageLandmarks.add(new Point(163, 148));
        selfieImageLandmarks.add(new Point(151, 148));
        selfieImageLandmarks.add(new Point(92, 221));
        selfieImageLandmarks.add(new Point(103, 214));
        selfieImageLandmarks.add(new Point(113, 209));
        selfieImageLandmarks.add(new Point(120, 211));
        selfieImageLandmarks.add(new Point(126, 209));
        selfieImageLandmarks.add(new Point(137, 213));
        selfieImageLandmarks.add(new Point(148, 220));
        selfieImageLandmarks.add(new Point(137, 226));
        selfieImageLandmarks.add(new Point(127, 228));
        selfieImageLandmarks.add(new Point(119, 229));
        selfieImageLandmarks.add(new Point(112, 229));
        selfieImageLandmarks.add(new Point(103, 227));
        selfieImageLandmarks.add(new Point(98, 221));
        selfieImageLandmarks.add(new Point(113, 218));
        selfieImageLandmarks.add(new Point(120, 218));
        selfieImageLandmarks.add(new Point(127, 217));
        selfieImageLandmarks.add(new Point(143, 220));
        selfieImageLandmarks.add(new Point(126, 218));
        selfieImageLandmarks.add(new Point(119, 218));
        selfieImageLandmarks.add(new Point(112, 218));


        hairModelImageLandmark.add(new Point(96, 366));
        hairModelImageLandmark.add(new Point(98, 425));
        hairModelImageLandmark.add(new Point(107, 486));
        hairModelImageLandmark.add(new Point(121, 547));
        hairModelImageLandmark.add(new Point(145, 602));
        hairModelImageLandmark.add(new Point(180, 651));
        hairModelImageLandmark.add(new Point(222, 698));
        hairModelImageLandmark.add(new Point(268, 734));
        hairModelImageLandmark.add(new Point(319, 745));
        hairModelImageLandmark.add(new Point(369, 733));
        hairModelImageLandmark.add(new Point(417, 695));
        hairModelImageLandmark.add(new Point(461, 649));
        hairModelImageLandmark.add(new Point(497, 599));
        hairModelImageLandmark.add(new Point(518, 543));
        hairModelImageLandmark.add(new Point(529, 482));
        hairModelImageLandmark.add(new Point(539, 423));
        hairModelImageLandmark.add(new Point(542, 362));
        hairModelImageLandmark.add(new Point(124, 320));
        hairModelImageLandmark.add(new Point(161, 303));
        hairModelImageLandmark.add(new Point(202, 303));
        hairModelImageLandmark.add(new Point(242, 316));
        hairModelImageLandmark.add(new Point(281, 337));
        hairModelImageLandmark.add(new Point(357, 339));
        hairModelImageLandmark.add(new Point(396, 318));
        hairModelImageLandmark.add(new Point(438, 303));
        hairModelImageLandmark.add(new Point(481, 302));
        hairModelImageLandmark.add(new Point(518, 322));
        hairModelImageLandmark.add(new Point(317, 383));
        hairModelImageLandmark.add(new Point(317, 427));
        hairModelImageLandmark.add(new Point(317, 473));
        hairModelImageLandmark.add(new Point(316, 518));
        hairModelImageLandmark.add(new Point(277, 537));
        hairModelImageLandmark.add(new Point(296, 544));
        hairModelImageLandmark.add(new Point(317, 552));
        hairModelImageLandmark.add(new Point(338, 544));
        hairModelImageLandmark.add(new Point(359, 537));
        hairModelImageLandmark.add(new Point(172, 371));
        hairModelImageLandmark.add(new Point(197, 358));
        hairModelImageLandmark.add(new Point(230, 360));
        hairModelImageLandmark.add(new Point(254, 383));
        hairModelImageLandmark.add(new Point(224, 389));
        hairModelImageLandmark.add(new Point(194, 388));
        hairModelImageLandmark.add(new Point(385, 383));
        hairModelImageLandmark.add(new Point(410, 360));
        hairModelImageLandmark.add(new Point(443, 358));
        hairModelImageLandmark.add(new Point(468, 373));
        hairModelImageLandmark.add(new Point(446, 390));
        hairModelImageLandmark.add(new Point(415, 390));
        hairModelImageLandmark.add(new Point(235, 607));
        hairModelImageLandmark.add(new Point(265, 595));
        hairModelImageLandmark.add(new Point(296, 585));
        hairModelImageLandmark.add(new Point(317, 593));
        hairModelImageLandmark.add(new Point(336, 586));
        hairModelImageLandmark.add(new Point(367, 597));
        hairModelImageLandmark.add(new Point(398, 610));
        hairModelImageLandmark.add(new Point(370, 648));
        hairModelImageLandmark.add(new Point(339, 664));
        hairModelImageLandmark.add(new Point(317, 665));
        hairModelImageLandmark.add(new Point(294, 662));
        hairModelImageLandmark.add(new Point(263, 645));
        hairModelImageLandmark.add(new Point(251, 611));
        hairModelImageLandmark.add(new Point(296, 612));
        hairModelImageLandmark.add(new Point(317, 616));
        hairModelImageLandmark.add(new Point(337, 613));
        hairModelImageLandmark.add(new Point(382, 612));
        hairModelImageLandmark.add(new Point(338, 620));
        hairModelImageLandmark.add(new Point(317, 622));
        hairModelImageLandmark.add(new Point(295, 619));


        transformationFromPoints(selfieImageLandmarks, hairModelImageLandmark);
        Mat procrustes = new Mat(2, 3, CV_64F);

        procrustes.put(0, 0, 2.87963694e+00);
        procrustes.put(0, 1, -2.23464350e-02);
        procrustes.put(0, 2, -2.21126139e+01);
        procrustes.put(1, 0, 2.23464350e-02);
        procrustes.put(1, 1, 2.87963694e+00);
        procrustes.put(1, 2, -3.04007747e+01);

//        System.out.println(procrustes.dump());

        Mat warpedSelfieImageMat = warp(selfieImageMat, procrustes, hairModelImageMat.size());
//        return warpedSelfieImageMat;

//        ArrayList<Point> sourcefaceLandmarks = getLandmarks(warpedSelfieImageBitmap);

        ArrayList<Point> sourcefaceLandmarks = new ArrayList<>();
        sourcefaceLandmarks.add(new Point(41, 137));
        sourcefaceLandmarks.add(new Point(42, 159));
        sourcefaceLandmarks.add(new Point(46, 181));
        sourcefaceLandmarks.add(new Point(51, 202));
        sourcefaceLandmarks.add(new Point(61, 221));
        sourcefaceLandmarks.add(new Point(73, 238));
        sourcefaceLandmarks.add(new Point(88, 253));
        sourcefaceLandmarks.add(new Point(104, 265));
        sourcefaceLandmarks.add(new Point(121, 268));
        sourcefaceLandmarks.add(new Point(138, 264));
        sourcefaceLandmarks.add(new Point(155, 251));
        sourcefaceLandmarks.add(new Point(170, 234));
        sourcefaceLandmarks.add(new Point(182, 217));
        sourcefaceLandmarks.add(new Point(190, 197));
        sourcefaceLandmarks.add(new Point(193, 175));
        sourcefaceLandmarks.add(new Point(197, 154));
        sourcefaceLandmarks.add(new Point(197, 133));
        sourcefaceLandmarks.add(new Point(52, 121));
        sourcefaceLandmarks.add(new Point(64, 115));
        sourcefaceLandmarks.add(new Point(79, 115));
        sourcefaceLandmarks.add(new Point(93, 119));
        sourcefaceLandmarks.add(new Point(106, 127));
        sourcefaceLandmarks.add(new Point(132, 128));
        sourcefaceLandmarks.add(new Point(146, 120));
        sourcefaceLandmarks.add(new Point(160, 115));
        sourcefaceLandmarks.add(new Point(175, 114));
        sourcefaceLandmarks.add(new Point(188, 120));
        sourcefaceLandmarks.add(new Point(119, 143));
        sourcefaceLandmarks.add(new Point(119, 158));
        sourcefaceLandmarks.add(new Point(119, 173));
        sourcefaceLandmarks.add(new Point(119, 189));
        sourcefaceLandmarks.add(new Point(105, 196));
        sourcefaceLandmarks.add(new Point(112, 198));
        sourcefaceLandmarks.add(new Point(119, 201));
        sourcefaceLandmarks.add(new Point(127, 198));
        sourcefaceLandmarks.add(new Point(134, 196));
        sourcefaceLandmarks.add(new Point(68, 139));
        sourcefaceLandmarks.add(new Point(77, 134));
        sourcefaceLandmarks.add(new Point(88, 135));
        sourcefaceLandmarks.add(new Point(97, 143));
        sourcefaceLandmarks.add(new Point(87, 145));
        sourcefaceLandmarks.add(new Point(76, 145));
        sourcefaceLandmarks.add(new Point(142, 143));
        sourcefaceLandmarks.add(new Point(150, 134));
        sourcefaceLandmarks.add(new Point(162, 133));
        sourcefaceLandmarks.add(new Point(171, 138));
        sourcefaceLandmarks.add(new Point(164, 144));
        sourcefaceLandmarks.add(new Point(152, 145));
        sourcefaceLandmarks.add(new Point(92, 221));
        sourcefaceLandmarks.add(new Point(102, 216));
        sourcefaceLandmarks.add(new Point(113, 213));
        sourcefaceLandmarks.add(new Point(119, 215));
        sourcefaceLandmarks.add(new Point(126, 213));
        sourcefaceLandmarks.add(new Point(137, 217));
        sourcefaceLandmarks.add(new Point(148, 221));
        sourcefaceLandmarks.add(new Point(137, 235));
        sourcefaceLandmarks.add(new Point(127, 240));
        sourcefaceLandmarks.add(new Point(120, 241));
        sourcefaceLandmarks.add(new Point(112, 240));
        sourcefaceLandmarks.add(new Point(101, 234));
        sourcefaceLandmarks.add(new Point(97, 222));
        sourcefaceLandmarks.add(new Point(113, 222));
        sourcefaceLandmarks.add(new Point(120, 223));
        sourcefaceLandmarks.add(new Point(126, 222));
        sourcefaceLandmarks.add(new Point(142, 222));
        sourcefaceLandmarks.add(new Point(126, 225));
        sourcefaceLandmarks.add(new Point(119, 225));
        sourcefaceLandmarks.add(new Point(112, 224));

        Point p17 = sourcefaceLandmarks.get(17);
        Point p19 = sourcefaceLandmarks.get(19);
        Point p24 = sourcefaceLandmarks.get(24);
        Point p26 = sourcefaceLandmarks.get(26);
        Point p8 = sourcefaceLandmarks.get(8);

        org.opencv.core.Point p17_Opencv = new org.opencv.core.Point(p17.x, p17.y);
        org.opencv.core.Point p19_Opencv = new org.opencv.core.Point(p19.x, p19.y);
        org.opencv.core.Point p24_Opencv = new org.opencv.core.Point(p24.x, p24.y);
        org.opencv.core.Point p26_Opencv = new org.opencv.core.Point(p26.x, p26.y);
        org.opencv.core.Point p8_Opencv = new org.opencv.core.Point(p8.x, p8.y);

        double r8_24 = Core.norm(new MatOfPoint(p8_Opencv), new MatOfPoint(p24_Opencv));
        double r8_17 = Core.norm(new MatOfPoint(p8_Opencv), new MatOfPoint(p17_Opencv));
        double r8_26 = Core.norm(new MatOfPoint(p8_Opencv), new MatOfPoint(p26_Opencv));
        double r8_19 = Core.norm(new MatOfPoint(p8_Opencv), new MatOfPoint(p19_Opencv));

        double r1 = r8_17 * (1 + (1 / 32f));
        double r2 = r8_19 * (1 + (1 / 8f));
        double r3 = r8_24 * (1 + (1 / 8f));
        double r4 = r8_26 * (1 + (1 / 32f));

        double p8x = p8.x;
        double p8y = p8.y;

        double p17x = p17.x;
        double p17y = p17.y;
        double theta1 = Math.atan(Math.abs((p17y - p8y) / (p17x - p8x)));

        double p19x = p19.x;
        double p19y = p19.y;
        double theta2 = Math.atan(Math.abs((p19y - p8y) / (p19x - p8x)));

        double p24x = p24.x;
        double p24y = p24.y;
        double theta3 = Math.atan(Math.abs((p24y - p8y) / (p24x - p8x)));

        double p26x = p26.x;
        double p26y = p26.y;
        double theta4 = Math.atan(Math.abs((p26y - p8y) / (p26x - p8x)));

        Point p68 = new Point(((int)(p8x - r2 * Math.cos(theta2))), ((int)(p8y - r2 * Math.sin(theta2))));
        Point p69 = new Point((int) (p8x + r3 * Math.cos(theta3)), (int)(p8y - r3 * Math.sin(theta3)));
        Point p70 = new Point((int) (p8x - r1 * Math.cos(theta1)), (int)(p8y - r1 * Math.sin(theta1)));
        Point p71 = new Point((int) (p8x + r4 * Math.cos(theta4)), (int)(p8y - r4 * Math.sin(theta4)));

        ArrayList<Point> sourcefaceLandmarksAug = new ArrayList<>(sourcefaceLandmarks);
        sourcefaceLandmarksAug.set(19, p68);
        sourcefaceLandmarksAug.set(24, p69);
        sourcefaceLandmarksAug.set(17, p70);
        sourcefaceLandmarksAug.set(26, p71);

        Mat combinedFaceMask = getFaceMask(warpedSelfieImageMat, sourcefaceLandmarksAug);
//        Mat combinedFaceMask = Mat.zeros(warpedSelfieImageMat.size(),
//                warpedSelfieImageMat.depth());

        Mat newLookMat = edgeRefinement(warpedSelfieImageMat, hairModelImageMat,
                combinedFaceMask, sourcefaceLandmarksAug);

//        warpedSelfieImageMat.copyTo(hairModelImage, combinedFaceMask);

//        System.out.println("" + warpedSelfieImageMat.channels() + " | " + hairModelImage.channels());

        return newLookMat;

    }

    private ArrayList<Point> getLandmarks(Bitmap image){



        ArrayList<Point> landmarks = new ArrayList<>();

        /*InputStream is = null;
        try {
            is = context.getAssets().open("hair_model_image.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(is);
            String imageTitle = context.getResources().getString(R.string.temp_image_title);

            FileUtils.SaveImageToPhone(bitmap, imageTitle, context);
            String rootFolder = Environment.getExternalStorageDirectory().toString();

            String imageInSD = String.format(context.getResources()
                    .getString(R.string.faceswap_temp_image_path_in_phone), rootFolder);

            imageInSD += imageTitle;

            Log.i(TAG, imageInSD);

        if (mFaceDet == null) {
            mFaceDet = new FaceDet(Constants.getFaceShapeModelPath());
        }

        final String targetPath = Constants.getFaceShapeModelPath();
        if (!new File(targetPath).exists()) {

            Log.d(TAG, "shape predictor found");
            FileUtils.copyFileFromRawToOthers(context, R.raw.shape_predictor_68_face_landmarks, targetPath);
        }else{
            Log.d(TAG, "shape predictor not found");
        }

        List<VisionDetRet> faceList = mFaceDet.detect(imageInSD);

        if(faceList.size() > 0) {
            //do something
            Log.i("GET LANDMARKS", faceList.size() + "");
            landmarks = faceList.get(0).getFaceLandmarks();
        } else {
            //do something else
            Log.i("GET LANDMARKS",  "nothing found");
        }

        //=======================================================//

        } catch (IOException e) {
            e.printStackTrace();

            Log.e("GET LANDMARKS",  "error getting landmark", e);
        }*/


        return landmarks;
    }

    /**
     @staticmethod
     def get_landmarks(image):
     """
     :param: image, a numpy array representation of the input image to find landmarks
     :type: numpy array
     :return: landmarks, a (68 x 2) matrix of coordinates, of special features in any image, for this instance a face.
     :type: matrix of dimension (68 x 2)
     """
     image_ = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
     detector = dlib.get_frontal_face_detector()
     rects = detector(image_, 1)
     predictor = dlib.shape_predictor(os.path.dirname(__file__)+"/shape_predictor_68_face_landmarks.dat")

     #68 x 2 matrix (numpy.matrixlib.defmatrix.matrix)
     landmarks = np.matrix([[int(p.x), int(p.y)] for p in predictor(image_, rects[0]).parts()])

     return landmarks
      * @param selfieImageLandmarks
     * @param hairModelImageLandmark
     */

    public Mat transformationFromPoints(ArrayList<Point> selfieImageLandmarks,
                                        ArrayList<Point> hairModelImageLandmark){


        org.opencv.core.Point selfieImage [] = new org.opencv.core.Point[selfieImageLandmarks.size()];
        for(int i = 0; i < selfieImageLandmarks.size(); i++){
            selfieImage[i] = new org.opencv.core.Point(selfieImageLandmarks.get(i).x,
                                                        selfieImageLandmarks.get(i).y);
        }

        org.opencv.core.Point hairModel [] = new org.opencv.core.Point[hairModelImageLandmark.size()];
        for(int i = 0; i < selfieImageLandmarks.size(); i++){
            hairModel[i] = new org.opencv.core.Point(hairModelImageLandmark.get(i).x,
                                                        hairModelImageLandmark.get(i).y);
        }

        MatOfPoint point1 = new MatOfPoint(selfieImage);
        MatOfPoint point2 = new MatOfPoint(hairModel);

        Scalar c1 = Core.mean(point1);
        Scalar c2 = Core.mean(point2);

        Core.subtract(point1, c1, point1);
        Core.subtract(point2, c2, point2);

        MatOfDouble mean1 = new MatOfDouble();
        MatOfDouble s1 = new MatOfDouble();
        Core.meanStdDev(point1, mean1, s1);
        Scalar stdScalar1 = new Scalar(s1.get(0, 0)[0]);


        MatOfDouble mean2 = new MatOfDouble();
        MatOfDouble s2 = new MatOfDouble();
        Core.meanStdDev(point2, mean2, s2);
        Scalar stdScalar2 = new Scalar(s2.get(0, 0)[0]);

        Core.divide(point1, stdScalar1, point1);
        Core.divide(point2, stdScalar2, point2);

//        point1.convertTo(point1, CV_32FC2);
//        point2.convertTo(point2, CV_32FC2);
//        Mat src = point1.t().mul(point2);
//
//        System.out.println("SOURCE " + src.dump());
        Mat src = new Mat();

        point1.convertTo(point1, CV_32FC2);
        point2.convertTo(point2, CV_32FC2);
        Core.gemm(point1.t(), point2, 1, new Mat(), 0, src);


        Mat w = new Mat();
        Mat u = new Mat();
        Mat vt = new Mat();

        Core.SVDecomp(src, w, u, vt);

        //do something
        return null;
    }

    public Mat warp(Mat selfieImageMat, Mat M, Size size){

        Mat temp = new Mat(size, CV_8UC3);

        warpAffine(selfieImageMat, temp, M, size, WARP_INVERSE_MAP
                , BORDER_TRANSPARENT, new Scalar(0));

        return temp;
    }

    public Mat getFaceMask(Mat image, ArrayList<Point> landmarks){

        List<org.opencv.core.Point> listOfPoints = new ArrayList<>();

        for(Point pt : landmarks)
            listOfPoints.add(new org.opencv.core.Point(pt.x, pt.y));

        org.opencv.core.Point [] convexPoints = new org.opencv.core.Point[landmarks.size()];
        for(int i = 0; i < landmarks.size(); i++){
            convexPoints[i] = new org.opencv.core.Point(landmarks.get(i).x, landmarks.get(i).y);
        }

        MatOfPoint matOfPoint = new MatOfPoint(convexPoints);
        MatOfInt matOfInt = new MatOfInt();

        for(int i = 0; i < landmarks.size(); i++){
            convexHull(matOfPoint, matOfInt);
        }


        MatOfPoint mopOut = new MatOfPoint();
        mopOut.create((int)matOfInt.size().height,1,CvType.CV_32SC2);

        for(int i = 0; i < matOfInt.size().height ; i++)
        {
            int index = (int)matOfInt.get(i, 0)[0];
            double[] point = new double[] {
                    matOfPoint.get(index, 0)[0], matOfPoint.get(index, 0)[1]
            };
            mopOut.put(i, 0, point);
        }

//        convexHull(matOfPoint, matOfInt);
/*

        List<Point> faceConvexHull = new ArrayList<>();
        faceConvexHull.add(new Point(197, 154));
        faceConvexHull.add(new Point(190, 197));
        faceConvexHull.add(new Point(182, 217));
        faceConvexHull.add(new Point(170, 234));
        faceConvexHull.add(new Point(155, 251));
        faceConvexHull.add(new Point(138, 264));
        faceConvexHull.add(new Point(121, 268));
        faceConvexHull.add(new Point(104, 265));
        faceConvexHull.add(new Point(88, 253));
        faceConvexHull.add(new Point(73, 238));
        faceConvexHull.add(new Point(61, 221));
        faceConvexHull.add(new Point(51, 202));
        faceConvexHull.add(new Point(46, 181));
        faceConvexHull.add(new Point(42, 159));
        faceConvexHull.add(new Point(41, 137));
        faceConvexHull.add(new Point(49, 116));
        faceConvexHull.add(new Point(73,  95));
        faceConvexHull.add(new Point(164,  95));
        faceConvexHull.add(new Point(190, 115));
        faceConvexHull.add(new Point(197, 133));

        org.opencv.core.Point [] pointArray = new org.opencv.core.Point[faceConvexHull.size()];

        Point pt;
        for(int i = 0; i < faceConvexHull.size(); i++){
            pt = faceConvexHull.get(i);
            pointArray[i] = new org.opencv.core.Point(pt.x, pt.y);
        }



        Mat faceMask =  Mat.zeros(image.size(), CV_8UC3);

        MatOfPoint points = new MatOfPoint(pointArray);


        fillConvexPoly(faceMask, points, new Scalar(255, 255, 255));
*/



        Mat faceMask =  Mat.zeros(image.size(), CV_8UC3);

        fillConvexPoly(faceMask, mopOut, new Scalar(255, 255, 255));

        System.out.println("HULL " + matOfInt.size());

        return faceMask;
    }


    public Mat edgeRefinement(Mat sourceImage, Mat destinationImage, Mat sourcefaceMask,
                                     ArrayList<Point> hairfaceLandMarks){

        ArrayList<MatOfPoint> points = new ArrayList<>();
        for(int i = 0; i < hairfaceLandMarks.size(); i++){
            points.add(i, new MatOfPoint(
                    new org.opencv.core.Point(hairfaceLandMarks.get(i).x,
                            hairfaceLandMarks.get(i).y)));

//            System.out.println(Arrays.toString(points.get(i).toArray()));
        }

        System.out.println("SORUCE IMAGE " + sourceImage.size() + " " +sourceImage.channels());
        System.out.println("DESTINATION IMAGE " + destinationImage.size() + " " + destinationImage.channels());
        System.out.println("SOURCE FACE MASK IMAGE " + sourcefaceMask.size() + " " + sourcefaceMask.channels());
        System.out.println("LANDMARKS " + hairfaceLandMarks.size());

//        org.opencv.core.Point center = faceCenter(points);
        org.opencv.core.Point center =
                new org.opencv.core.Point(119, 182);

        Mat edgeBlended = new Mat(destinationImage.size(), CvType.CV_8UC3);
        Photo.seamlessClone(sourceImage, destinationImage, sourcefaceMask,
                center, edgeBlended, NORMAL_CLONE);

        System.out.println(edgeBlended.height()+ " " + edgeBlended.width()+ " " + edgeBlended.channels());
//        edgeBlended = sourcefaceMask;

        return edgeBlended;
    }

    public org.opencv.core.Point faceCenter(ArrayList<MatOfPoint> hairfaceLandmarks){

        ArrayList<MatOfInt> hairfaceConvexHull = new ArrayList<>(hairfaceLandmarks.size());
        ArrayList<Rect> rectVertices = new ArrayList<>(hairfaceLandmarks.size());


        for(int i = 0; i < hairfaceLandmarks.size(); i++){

            hairfaceConvexHull.add(i, new MatOfInt());
            convexHull(hairfaceLandmarks.get(i), hairfaceConvexHull.get(i));
            rectVertices.add(i, boundingRect(hairfaceLandmarks.get(i)));
        }

        org.opencv.core.Point point = new org.opencv.core.Point(119, 182);

        return point;
    }
    /**
     @staticmethod
     def face_center(hairface_landmarks):
     """
     :param hairface_landmarks: np array, each row contains [x,y] co-ordinates of landmarks on the input image
     :return: (center-x, center-y) co-ordinates of the center of the image.
     """
     hairface_convex_hull = cv2.convexHull(hairface_landmarks)[:, 0]

     rect_vertices = cv2.boundingRect(np.float32(hairface_convex_hull))
     center = (int(rect_vertices[0]) + int(rect_vertices[2] / 2), int(rect_vertices[1]) + int(rect_vertices[3] / 2))

     return center
     */

}
