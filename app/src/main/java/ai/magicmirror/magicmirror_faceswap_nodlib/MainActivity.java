package ai.magicmirror.magicmirror_faceswap_nodlib;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import ai.magicmirror.magicmirror_faceswap_nodlib.HairSwap.FaceSwap;
import ai.magicmirror.magicmirror_faceswap_nodlib.HairSwap.FaceSwap_mat;
import ai.magicmirror.magicmirror_faceswap_nodlib.HairSwap.FaceSwap_test;

public class MainActivity extends AppCompatActivity {

    static{ System.loadLibrary("opencv_java3"); }

    private InputStream is;

    ImageView image;
    private final int PICK_IMAGE_ID = 200;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        image = findViewById(R.id.image_view);
/*

        try {
            is = getAssets().open("hair_model_image.jpg");
            Bitmap hairModelBitmap = BitmapFactory.decodeStream(is);

            is = getAssets().open("selfie_image.jpg");
            Bitmap selfieImage = BitmapFactory.decodeStream(is);

            Mat selfieMat = new Mat();
            Utils.bitmapToMat(selfieImage, selfieMat);
            Imgproc.cvtColor(selfieMat, selfieMat, Imgproc.COLOR_BGRA2BGR);

            Mat hairModelMat = new Mat();
            Utils.bitmapToMat(hairModelBitmap, hairModelMat);
            Imgproc.cvtColor(hairModelMat, hairModelMat, Imgproc.COLOR_BGRA2BGR);

   */
/*         Mat selfieMat = Utils.loadResource(MainActivity.this,
                    R.drawable.selfie_image, Imgcodecs.CV_IMWRITE_JPEG_OPTIMIZE);

            Mat hairModelMat = Utils.loadResource(MainActivity.this,
                    R.drawable.hair_model_image, Imgcodecs.CV_IMWRITE_JPEG_OPTIMIZE);*//*


*/
/*            Bitmap selfie = Bitmap.createBitmap(selfieMat.width(), selfieMat.height(),
                    Bitmap.Config.ARGB_8888);

            Bitmap hairMoldel = Bitmap.createBitmap(hairModelMat.width(), hairModelMat.height(),
                    Bitmap.Config.ARGB_8888);

            Utils.matToBitmap(selfieMat, selfie);
            Utils.matToBitmap(hairModelMat, hairMoldel);*//*


//            System.out.println("IN MAIN ==> " + m.channels());

//        Bitmap faceTest = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888);
            */
/*image.setImageBitmap(FaceSwap.getInstance(getApplicationContext())
                    .swap(selfieImage, hairModelBitmap));*//*

//



                        Mat newLook = FaceSwap_mat.getInstance(getApplicationContext())
                    .swap(hairModelMat, selfieMat);

            Bitmap faceTest = Bitmap.createBitmap(newLook.width(), newLook.height(),
                    Bitmap.Config.ARGB_8888);

            Utils.matToBitmap(newLook, faceTest);

            image.setImageBitmap(faceTest);



        } catch (IOException e) {
            e.printStackTrace();
        }

        Log.d("MAIN ACTIVITY", "Done");
*/

//        ImagePicker.setMinQuality(600, 600);

        findViewById(R.id.button)
                .setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        Intent chooseImageIntent = ImagePicker.getPickImageIntent(MainActivity.this);
                        startActivityForResult(chooseImageIntent, PICK_IMAGE_ID);
                    }
                });


    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        switch (requestCode) {
            case PICK_IMAGE_ID:
                Bitmap bitmap = ImagePicker.getImageFromResult(this, resultCode, data);
                // TODO use bitmap

                image.setImageBitmap(bitmap);
                break;
            default:
                super.onActivityResult(requestCode, resultCode, data);
                break;

        }

    }

}
