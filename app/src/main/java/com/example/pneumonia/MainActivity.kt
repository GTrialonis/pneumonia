package com.example.pneumonia

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val buttonOpenCamera = findViewById<Button>(R.id.button)
        val buttonOpenGallery = findViewById<Button>(R.id.button2)

        buttonOpenCamera.setOnClickListener{
            Intent(this, Camera_CheckPneumonia::class.java).also{
                startActivity(it)
            }
        }
        buttonOpenGallery.setOnClickListener{
            Intent(this, Gallery_ChestXray::class.java).also{
                startActivity(it)
            }
        }
    }
}