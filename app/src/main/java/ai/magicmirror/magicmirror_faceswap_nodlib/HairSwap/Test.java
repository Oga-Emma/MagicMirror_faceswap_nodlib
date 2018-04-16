package ai.magicmirror.magicmirror_faceswap_nodlib.HairSwap;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

import java.util.ArrayList;

/**
 * Created by seven on 3/23/18.
 */

public class Test {


    public static void swap(Mat src, Mat dest) {
        // Create a rough mask around the airplane.

        Mat src_mask = Mat.zeros(src.size(), src.type());

        // Define the mask as a closed polygon
     /*   Point poly[ 1][7];
        poly[0][0] = Point(4, 80);
        poly[0][1] = Point(30, 54);
        poly[0][2] = Point(151, 63);
        poly[0][3] = Point(254, 37);
        poly[0][4] = Point(298, 90);
        poly[0][5] = Point(272, 134);
        poly[0][6] = Point(43, 122);*/

        ArrayList<MatOfPoint> list = new ArrayList<>();
        list.add(new MatOfPoint(new Point(4, 80)));
        list.add(new MatOfPoint(new Point(30, 54)));
        list.add(new MatOfPoint(new Point(151, 63)));
        list.add(new MatOfPoint(new Point(254, 37)));
        list.add(new MatOfPoint(new Point(298, 90)));
        list.add(new MatOfPoint(new Point(272, 134)));
        list.add(new MatOfPoint(new Point(43, 122)));

        Imgproc.fillPoly(src_mask, list, new Scalar(255, 255, 255));

/*const Point * polygons[1] = {poly[0]};
        int num_points[] = {7};*/

// Create mask by filling the polygon

//        fillPoly(src_mask, polygons, num_points, 1, Scalar(255, 255, 255));



        // The location of the center of the src in the dst
        Point center = new Point(800, 100);

        // Seamlessly clone src into dst and put the results in output
        Mat output = new Mat();
        Photo.seamlessClone(src, dest, src_mask, center, output, Photo.NORMAL_CLONE);

    }

}
