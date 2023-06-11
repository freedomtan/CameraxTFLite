package com.mediatek.cameraxtflite

import android.Manifest
import android.content.ContentValues.TAG
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.mediatek.cameraxtflite.databinding.ActivityMainBinding
import org.tensorflow.lite.classify.MyClassifierModel
import org.tensorflow.lite.support.model.Model
import java.util.concurrent.Executors

// This is an arbitrary number we are using to keep track of the permission
// request. Where an app has multiple context for requesting permission,
// this can help differentiate the different contexts.
private const val REQUEST_CODE_PERMISSIONS = 10

// This is an array of all the permission specified in the manifest.
private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
/*
fun Image.toBitmap(): Bitmap {
    val yBuffer = planes[0].buffer // Y
    val uBuffer = planes[1].buffer // U
    val vBuffer = planes[2].buffer // V

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    // U and V are swapped
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 50, out)
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}*/

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private var imageCapture: ImageCapture? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // cameraView = findViewById(R.id.camera_view)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Request camera permissions
        if (allPermissionsGranted()) {
            // cameraView.post { startCamera() }
            startCamera()
        } else {
            /*
                ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )*/
            requestPermissions()
        }

        /*
        // Every time the provided texture view changes, recompute layout
        cameraView.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ ->
            updateTransform()
        }*/

        latencyText = findViewById(R.id.latency)
        top1Text = findViewById(R.id.top1)
    }

    // Add this after onCreate

    private val executor = Executors.newSingleThreadExecutor()

    // private lateinit var cameraView: TextureView
    private lateinit var latencyText: TextView
    private lateinit var top1Text: TextView

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.cameraView.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .build()
                .also {
                    it.setAnalyzer(executor, TfLiteAnalyzer(this@MainActivity))
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer
                )

            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun requestPermissions() {}

    /**
     * Check if all permission specified in the manifest have been granted
     */
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    private class TfLiteAnalyzer(outer: MainActivity) : ImageAnalysis.Analyzer {
        private val mainActivity = outer
        override fun analyze(image: ImageProxy) {
            val bitmap = image.toBitmap()
            val myImageClassifier =
                MyClassifierModel(mainActivity.getApplicationContext(), Model.Device.NNAPI, 4)

            val inputs = myImageClassifier.createInputs()
            inputs.loadImage(bitmap)

            val startTimestamp = System.nanoTime()
            val outputs = myImageClassifier.run(inputs)
            val stopTimestamp = System.nanoTime()

            val labeledProbability = outputs.getProbability()

            var maxEntry: Map.Entry<String, Float>? = null
            for (entry in labeledProbability.entries) {
                if (maxEntry == null || entry.value.compareTo(maxEntry.value) > 0) {
                    maxEntry = entry
                }
            }

            mainActivity.runOnUiThread(Runnable {
                mainActivity.top1Text.text = "top-1: " + maxEntry.toString()
                mainActivity.latencyText.text =
                    "latency: " + (stopTimestamp - startTimestamp) / 1000000.0 + " ms"
            })

            image.close()
        }
    }
}