﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Assets.Scripts.HackedSceneScripts
{
    class CameraController : MonoBehaviour
    {
        public GameObject player;

        // The distance in the x-z plane to the target
        public float distance = 12.0f;
        // the height we want the camera to be above the target
        public float height = 8.0f;
        public float heightDamping = 2.0f;
        public float rotationDamping = 3.0f;
        float wantedRotationAngle;
        float wantedHeight;
        float currentRotationAngle;
        float currentHeight;
        Quaternion currentRotation;
        // Start is called before the first frame update
        void Start()
        {
        }

        // Update is called once per frame
        void Update()
        {
            // Calculate the current rotation angles
            wantedRotationAngle = player.transform.eulerAngles.y;
            wantedHeight = player.transform.position.y + height;
            currentRotationAngle = transform.eulerAngles.y;
            currentHeight = transform.position.y;
            // Damp the rotation around the y-axis
            currentRotationAngle = Mathf.LerpAngle(currentRotationAngle, wantedRotationAngle, rotationDamping * Time.deltaTime);
            // Damp the height
            currentHeight = Mathf.Lerp(currentHeight, wantedHeight, heightDamping * Time.deltaTime);
            // Convert the angle into a rotation
            currentRotation = Quaternion.Euler(0, currentRotationAngle, 0);
            // Set the position of the camera on the x-z plane to:
            // distance meters behind the target
            transform.position = player.transform.position;
            transform.position -= currentRotation * Vector3.forward * distance;
            // Set the height of the camera
            transform.position = new Vector3(transform.position.x, currentHeight, transform.position.z);
            // Always look at the target
            transform.LookAt(player.transform);
            
        }
    }
}
