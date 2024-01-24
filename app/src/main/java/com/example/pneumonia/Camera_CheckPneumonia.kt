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
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat.checkSelfPermission
import androidx.core.app.ActivityCompat.requestPermissions
import com.example.pneumonia.ml.ChestxrayModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.roundToInt


@Suppress("DEPRECATION")
class Camera_CheckPneumonia: AppCompatActivity() {

    companion object {
        private const val CAMERA_REQUEST_CODE = 1
    }
    private lateinit var result: TextView
    private lateinit var imageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera_check_mole)

        // Load the TensorFlow Lite model and check its input shape
        val tflite = Interpreter(loadModelFile("chestXray_model.tflite"))
        val inputShape = tflite.getInputTensor(0).shape()
        Log.d("ModelInputShape", "Model expects input shape: ${inputShape.contentToString()}")

        // Close the interpreter to release resources
        tflite.close()

        result = findViewById(R.id.result)
        val btnBack = findViewById<Button>(R.id.btnBack)
        btnBack.setOnClickListener{
            finish()
        }
        imageView = findViewById(R.id.imageView)
        imageView.setOnClickListener{
            pickPhoto()
        }

    }

    private fun pickPhoto() {
        if (checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_REQUEST_CODE)
        } else {
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(cameraIntent, CAMERA_REQUEST_CODE)
        }
    }


    // Define loadModelFile outside of onCreate
    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun classifyImage(image: Bitmap) {
        // Convert the image to grayscale
        val grayscaleBitmap = convertToGrayscale(image)
        // Adjust the image for inference (contrast, brightness, etc.)
        val adjustedImage = adjustImageForInference(grayscaleBitmap)
        // The above two lines are new

        // Resize the adjusted image to the size expected by the model
        val resizedImage = Bitmap.createScaledBitmap(adjustedImage, 150, 150, true)
        // Load the model
        val model = ChestxrayModel.newInstance(applicationContext)

        // Initialize ByteBuffer for model input (4 bytes per float, 1 channel per pixel for grayscale)
        val byteBuffer = ByteBuffer.allocateDirect(4 * 150 * 150)
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
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 150, 150, 1), DataType.FLOAT32)
        inputFeature0.loadBuffer(byteBuffer)

        // Run inference and obtain the result
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val confidences: FloatArray = outputFeature0.floatArray

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

        // Clean up model resources
        model.close()
    }

    private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        val grayscaleBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayscaleBitmap)
        val paint = Paint()
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f)
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
    

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        // Check if the result is from the camera
        if (requestCode == CAMERA_REQUEST_CODE && resultCode == RESULT_OK) {
            val imageBitmap = data?.extras?.get("data") as Bitmap
            imageView.setImageBitmap(imageBitmap)

            // Call classifyImage with the image captured from the camera
            classifyImage(imageBitmap)
        }
        // You can remove the 'else if' part that checks for GALLERY_REQUEST_CODE
    }


}
