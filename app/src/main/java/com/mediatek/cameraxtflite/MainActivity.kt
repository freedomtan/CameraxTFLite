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

// This is an array of all the permission specified in the manifest.
private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private var imageCapture: ImageCapture? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }

        latencyText = viewBinding.latency
        top1Text = viewBinding.top1
    }


    private val executor = Executors.newSingleThreadExecutor()
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
                MyClassifierModel(mainActivity.applicationContext, Model.Device.NNAPI, 4)

            val inputs = myImageClassifier.createInputs()
            inputs.loadImage(bitmap)

            val startTimestamp = System.nanoTime()
            val outputs = myImageClassifier.run(inputs)
            val stopTimestamp = System.nanoTime()

            val labeledProbability = outputs.probability

            var maxEntry: Map.Entry<String, Float>? = null
            for (entry in labeledProbability.entries) {
                if (maxEntry == null || entry.value.compareTo(maxEntry.value) > 0) {
                    maxEntry = entry
                }
            }

            mainActivity.runOnUiThread {
                mainActivity.top1Text.text = "top-1: " + maxEntry.toString()
                mainActivity.latencyText.text =
                    "latency: " + (stopTimestamp - startTimestamp) / 1000000.0 + " ms"
            }

            image.close()

        }
}