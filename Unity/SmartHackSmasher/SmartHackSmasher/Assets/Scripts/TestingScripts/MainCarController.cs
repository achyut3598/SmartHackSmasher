using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MainCarController : MonoBehaviour
{
    public float speed = 40.0f;
    private float turnSpeed = 45.0f;
    public float distanceTraveled = 0f;
    public float speedToAccelerateTo = 60f;
    public float speedToStopAt = 0f;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        transform.Translate(Vector3.forward * Time.deltaTime * speed);//Moves Forward based on Verticl Input
        distanceTraveled += (speed * Time.timeScale);
        if (distanceTraveled > 26000f)
        {
            speed = speedToAccelerateTo;
        }
        if (distanceTraveled > 40000f)
        {
            speed = speedToStopAt;
        }
        //transform.Rotate(Vector3.up, Time.deltaTime * turnSpeed);//Rotates based on Horizontal Input
    }

    void PauseGame()
    {
        Time.timeScale = 0;
    }

    void ResumeGame()
    {
        Time.timeScale = 1;
    }
}
