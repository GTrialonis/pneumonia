package com.example.pneumonia

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.pneumonia.ml.ChestxrayModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

class Gallery_ChestXray : AppCompatActivity() {

    private lateinit var result: TextView
    private lateinit var imageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera_check_mole)

        result = findViewById(R.id.result)
        val btnBack = findViewById<Button>(R.id.btnBack)
        btnBack.setOnClickListener{
            finish()
        }
        imageView = findViewById(R.id.imageView)
        imageView.setOnClickListener{
            selectPhoto()
        }
    }

    private fun selectPhoto() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            // Permission is not granted
            // Should we show an explanation?
            if (ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.READ_EXTERNAL_STORAGE)) {
                // Show an explanation to the user *asynchronously* -- don't block
                // this thread waiting for the user's response! After the user
                // sees the explanation, try again to request the permission.
                // You can show a dialog or some other custom UI here with explanation
            } else {
                // No explanation needed; request the permission
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE), 1)
            }
        } else {
            // Permission has already been granted
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, 2)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            1 -> {
                // If request is cancelled, the result arrays are empty.
                if ((grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
                    // Permission was granted, yay! Do the
                    // contacts-related task you need to do.
                    selectPhoto() // Or directly open the gallery if permission is granted
                } else {
                    // Permission denied, boo! Disable the
                    // functionality that depends on this permission.
                    Toast.makeText(this, "Permission denied to read your External storage", Toast.LENGTH_SHORT).show()
                }
                return
            }
            // Other 'when' lines to check for other
            // permissions this app might request.
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == 2 && resultCode == RESULT_OK) {
            data?.data?.let { imageUri ->
                imageView.setImageURI(imageUri)
                val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, imageUri)
                classifyImage(bitmap)
            }
        }
    }

    private fun classifyImage(bitmap: Bitmap) {
        // Convert the image to grayscale
        val grayscaleBitmap = convertToGrayscale(bitmap)
        // Adjust the image for inference (contrast, brightness, etc.)
        val adjustedImage = adjustImageForInference(grayscaleBitmap)
        // The above two lines are new
        // Convert the image to grayscale and resize it to the size expected by the model
        // val grayscaleBitmap = convertToGrayscale(bitmap)
        val resizedImage = Bitmap.createScaledBitmap(grayscaleBitmap, 150, 150, true)

        // Load the model
        val model = ChestxrayModel.newInstance(applicationContext)


        // Initialize ByteBuffer for model input (4 bytes per float, 1 channel per pixel for grayscale)
        val byteBuffer = ByteBuffer.allocateDirect(4 * resizedImage.width * resizedImage.height)
        byteBuffer.order(ByteOrder.nativeOrder())

        // Fill the ByteBuffer with pixel values, normalized to [0,1]
        val intValues = IntArray(150 * 150)
        resizedImage.getPixels(
            intValues,
            0,
            resizedImage.width,
            0,
            0,
            resizedImage.width,
            resizedImage.height
        )
        // ----------- the below is new in relation to previous of 20/1/24------
        for (pixelValue in intValues) {
            val value = (pixelValue and 0xFF) / 255f // Use red channel value for grayscale
            byteBuffer.putFloat(value)
        }
        //----------------------------------------------------------------------

        // Prepare the model input with the ByteBuffer
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, resizedImage.width, resizedImage.height, 1), DataType.FLOAT32)
        inputFeature0.loadBuffer(byteBuffer)

        // Run inference and obtain the result
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val confidences: FloatArray = outputFeature0.floatArray
        //-----------------------------------------------------------------------------
        // Assuming confidences[0] is the probability of class 1 ("Pneumonia")
        val pneumoniaProbability = confidences[0]
        val predictionText = if (pneumoniaProbability >= 0.65) "PNEUMONIA" else "NORMAL"
        val probabilityText = if (pneumoniaProbability >= 0.65) {
            // Directly show the probability of Pneumonia
            "Probability: ${(pneumoniaProbability * 100).roundToInt()}%"
        } else {
            // Directly show the probability of Normal
            "Probability: ${((1 - pneumoniaProbability) * 100).roundToInt()}%"
        }
        result.text = "Prediction: $predictionText\n$probabilityText"
        //----------------------------------------------------------------------------

        // Clean up model resources
        model.close()
    }


    private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        val grayscaleBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayscaleBitmap)
        val paint = Paint()
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f) // Set to grayscale
        val filter = ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = filter
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        return grayscaleBitmap
    }

    fun adjustImageForInference(bitmap: Bitmap): Bitmap {
        // Create a mutable copy of the bitmap
        val adjustedBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(adjustedBitmap)
        val paint = Paint()

        // Adjust contrast (1.0 = no change, <1.0 = decrease, >1.0 = increase)
        val contrast = 1.8f

        // Adjust brightness (-255 to 255)
        val brightness = 50

        // Prepare the color matrix to adjust contrast and brightness
        val colorMatrix = ColorMatrix(floatArrayOf(
            contrast, 0f, 0f, 0f, brightness.toFloat(),
            0f, contrast, 0f, 0f, brightness.toFloat(),
            0f, 0f, contrast, 0f, brightness.toFloat(),
            0f, 0f, 0f, 1f, 0f
        ))

        paint.colorFilter = ColorMatrixColorFilter(colorMatrix)
        canvas.drawBitmap(bitmap, 0f, 0f, paint)

        return adjustedBitmap
    }
}
