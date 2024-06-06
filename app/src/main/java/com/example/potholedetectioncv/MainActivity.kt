package com.example.potholedetectioncv

import android.content.Context
import androidx.camera.core.CameraSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine
import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.location.Location
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.Column
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.app.ActivityCompat
import com.example.potholedetectioncv.ml.BestFp16Meta
import com.example.potholedetectioncv.ui.theme.PotholeDetectionCVTheme
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import org.tensorflow.lite.support.image.TensorImage
import java.text.SimpleDateFormat
import java.util.Date
import kotlin.math.PI
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import android.os.Looper
import android.view.WindowManager
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemGesturesPadding
import androidx.compose.material3.Slider
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.potholedetectioncv.ml.BestInt8Meta
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationRequest.Builder.IMPLICIT_MIN_UPDATE_INTERVAL
import kotlin.math.round


// const val THRESHOLD = 0.7
var THRESHOLD: Float by mutableFloatStateOf(0.4f)
const val MAX_WARNING_DISTANCE = 100
const val MIN_DISTANCE_BETWEEN_POTHOLES = 25

var potholesFound: Int by mutableIntStateOf(0)

private lateinit var fusedLocationClient: FusedLocationProviderClient

@SuppressLint("SimpleDateFormat")
val formatter = SimpleDateFormat("dd/MM/yyyy HH:mm:ss")
var date = Date()

var lastKnownLocation by mutableStateOf(Pair(0.0, 0.0))
var locationLastUpdated: String by mutableStateOf(formatter.format(date))

var potHoleFound by mutableStateOf("")

var potHoles: MutableList<Pair<Double, Double>> = mutableListOf()
var distanceToNearest: Double by mutableDoubleStateOf(Double.MAX_VALUE)


fun toRadians(n: Double): Double {
    return n * 2 * PI / 360
}

fun haversine(a: Pair<Double, Double>, b: Pair<Double, Double>): Double{
    return 2*6371000* asin((sin(toRadians(a.first - b.first)).pow(2)+ cos(
        toRadians(a.first)
    ) * cos(
        toRadians(b.first)
    ) * sin(toRadians((a.second - b.second) /2)).pow(2)).pow(0.5))
}

@RequiresApi(Build.VERSION_CODES.Q)
@SuppressLint("RestrictedApi")
@Composable
fun CameraPreviewScreen(alertFunc: (Boolean) -> Unit) {
    val lensFacing = CameraSelector.LENS_FACING_BACK
    val lifecycleOwner = LocalLifecycleOwner.current
    val context = LocalContext.current
    val preview = androidx.camera.core.Preview.Builder().build()
    val previewView = remember {
        PreviewView(context)
    }
    previewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE

    // val model = BestFp16Meta.newInstance(context)
    val model = BestInt8Meta.newInstance(context)

    previewView.setFrameUpdateListener(
        {
            val bitmap = previewView.bitmap
            bitmap?.config = Bitmap.Config.ARGB_8888
            //println("${bitmap?.width}, ${bitmap?.height}")

            // Creates inputs for reference.
            val image = TensorImage.fromBitmap(bitmap)
            // Runs model inference and gets result.
            val outputs = model.process(image)

            // Releases model resources if no longer used.
            // model.close()

            val inImageLocations: MutableList<List<Float>> = mutableListOf()
            // println(outputs)
            for (output in outputs.locationAsTensorBuffer.floatArray.toList().chunked(6)) {
                // println(output)
                if (output[4] >= THRESHOLD){
                    inImageLocations.add(output)
                }
            }
            println(inImageLocations.size)
            potholesFound = inImageLocations.size

            if (potholesFound > 0){
                alertFunc(true)
            }

        }, PreviewView.OnFrameUpdateListener {})


    val cameraxSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
    LaunchedEffect(lensFacing) {
        val cameraProvider = context.getCameraProvider()
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(lifecycleOwner, cameraxSelector, preview)
        preview.setSurfaceProvider(previewView.surfaceProvider)
    }
    AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
}

private suspend fun Context.getCameraProvider(): ProcessCameraProvider =
    suspendCoroutine { continuation ->
        ProcessCameraProvider.getInstance(this).also { cameraProvider ->
            cameraProvider.addListener({
                continuation.resume(cameraProvider.get())
            }, ContextCompat.getMainExecutor(this))
        }
    }

@Composable
fun Title(modifier: Modifier = Modifier) {
    Text(
        text = "Pothole Detection CV",
        modifier = modifier,
        fontSize = 30.sp
    )
}

@Composable
fun LocationText(modifier: Modifier = Modifier) {
    Text(
        text = "Current location is: $lastKnownLocation",
        modifier = modifier
    )
}

@Composable
fun LocationUpdateText(modifier: Modifier = Modifier) {
    Text(
        text = "Current location last updated: $locationLastUpdated",
        modifier = modifier
    )
}

@Composable
fun PotholesFoundSummaryText(modifier: Modifier = Modifier) {
    Text(
        text = potHoleFound,
        modifier = modifier
    )
    Text(
        text = "Number of potholes that have been found : ${potHoles.size}",
        modifier = modifier
    )
}

@Composable
fun PotholeWarningText(modifier: Modifier = Modifier){
    var redVal = 0
    if (distanceToNearest < MAX_WARNING_DISTANCE){
        redVal = 255-(distanceToNearest/ MAX_WARNING_DISTANCE * 255).toInt()
    }
    val modifier2 = modifier.background(color = Color(255, 0, 0, redVal))
    Text(
        text = "Distance to the nearest pothole : ${round(distanceToNearest)}m, $redVal",
        modifier = modifier2
    )
}

@Composable
fun ThresholdSlider(){
    Row {
        Slider(
            value = THRESHOLD,
            onValueChange = { THRESHOLD = it },
            modifier = Modifier.fillMaxWidth(0.6f).padding(15.dp, 0.dp)
        )
        Text(text = "Threshold: $THRESHOLD")
    }
}

@Composable
fun PotholesFoundText(modifier: Modifier = Modifier){
    Text(
        text = "$potholesFound were found in the image",
        modifier = modifier
    )
}

class MainActivity : ComponentActivity() {

    private lateinit var locationCallback: LocationCallback
    private lateinit var locationRequest: LocationRequest

    private var requestingLocationUpdates: Boolean = true

    @RequiresApi(Build.VERSION_CODES.Q)
    private val cameraPermissionRequest =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                setCameraPreview()
            } else {
                // Camera permission denied
            }
        }


    @RequiresApi(Build.VERSION_CODES.Q)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        when (PackageManager.PERMISSION_GRANTED) {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) -> {
                setCameraPreview()
            }
            else -> {
                cameraPermissionRequest.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun getLocationFunc(potholeFound: Boolean = false) {
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return
        }
        fusedLocationClient.lastLocation // this uses the last location stored
            //fusedLocationClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null) // this requests a current location but takes time
            .addOnSuccessListener { location: Location? ->
                // Got last known location. In some rare situations this can be null.
                println("location is $location")
                if (location != null) {
                    lastKnownLocation = Pair(location.latitude, location.longitude)
                    date = Date()
                    locationLastUpdated = formatter.format(date)
                    if (potholeFound) {
                        potHoleFound = "Pothole found at: $lastKnownLocation"
                        ToneGenerator(AudioManager.STREAM_MUSIC, 100).startTone(ToneGenerator.TONE_PROP_BEEP, 200)
                        var alreadyFound = false
                        loop@ for (item in potHoles) {
                            val distance = haversine(item, lastKnownLocation)
                            if (distance < MIN_DISTANCE_BETWEEN_POTHOLES) {
                                alreadyFound = true
                                break@loop
                            }
                        }
                        if (!alreadyFound) {
                            potHoles.add(lastKnownLocation)
                        }
                        distanceToNearest = 0.0
                    } else {
                        distanceToNearest = Double.MAX_VALUE
                        for (item in potHoles){
                            val distance = haversine(item, lastKnownLocation)
                            if (distance < distanceToNearest){
                                distanceToNearest = distance
                            }
                        }
                    }
                }
            }
        return
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun setCameraPreview() {
        setContent {
            PotholeDetectionCVTheme {
                Column {
                    Title(modifier = Modifier.padding(30.dp))
                    LocationText(modifier = Modifier.padding(6.dp))
                    LocationUpdateText(modifier = Modifier.padding(6.dp))
                    PotholesFoundSummaryText(modifier = Modifier.padding(6.dp))
                    PotholeWarningText(modifier = Modifier.padding(6.dp))
                    Surface(
                        modifier = Modifier.fillMaxSize(0.7F),
                        color = MaterialTheme.colorScheme.background
                    ) {
                        CameraPreviewScreen(alertFunc = ::getLocationFunc)
                    }
                    ThresholdSlider()
                    PotholesFoundText(modifier = Modifier.padding(6.dp))
                }
            }
        }

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        locationCallback = object : LocationCallback() {
            override fun onLocationResult(p0: LocationResult) {
                super.onLocationResult(p0)
                getLocationFunc()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (requestingLocationUpdates) startLocationUpdates()
    }

    private fun startLocationUpdates() {
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return
        }

        locationRequest = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 500)
            .apply {
                setWaitForAccurateLocation(true)
                setMinUpdateIntervalMillis(IMPLICIT_MIN_UPDATE_INTERVAL)
            }.build()

        fusedLocationClient.requestLocationUpdates(
            locationRequest,
            locationCallback,
            Looper.getMainLooper()
        )
    }

    override fun onPause() {
        super.onPause()
        stopLocationUpdates()
    }

    private fun stopLocationUpdates() {
        fusedLocationClient.removeLocationUpdates(locationCallback)
    }
}