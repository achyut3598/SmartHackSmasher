using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RoadSignPlayer : MonoBehaviour
{
    public float speed = 20.0f;
    public bool isHackedCar = false;
    public float timeSinceStop = 0f;
    private Rigidbody rigidbody;
    private float maxSpeed = 0f;

    public enum carBehavior { Straight, SpeedUp, OnRamp, Complete };
    public carBehavior currentCarBehavior;


    // Start is called before the first frame update
    void Start()
    {
        currentCarBehavior = carBehavior.Straight;
        rigidbody = GetComponent<Rigidbody>();
        if (isHackedCar)
        {
            maxSpeed = 40f;
        }
        else
        {
            maxSpeed = 30f;
        }
    }

    void Update()
    {
        switch (currentCarBehavior)
        {
            case carBehavior.Straight:
                transform.Translate(Vector3.forward * Time.deltaTime * speed);//Moves Forward based on Verticl Input
                break;
            case carBehavior.SpeedUp:
                if (Time.timeScale >0)
                {
                    speed *= 1.05f;
                }
                if (speed > maxSpeed)
                {
                    speed = maxSpeed;
                }
                transform.Translate(Vector3.forward * Time.deltaTime * speed);//Moves Forward
                break;
            case carBehavior.OnRamp:
                timeSinceStop += (Time.deltaTime * Time.timeScale);
                if (isHackedCar)
                {
                    if (timeSinceStop > 1.75f)
                    {
                        currentCarBehavior = carBehavior.Complete;
                    }
                    transform.Translate(Vector3.forward * Time.deltaTime * speed * .85f);//Moves Forward based on Verticl Input
                }
                else
                {
                    transform.Translate(Vector3.forward * Time.deltaTime * speed*.75f);//Moves Forward based on Verticl Input
                    if (timeSinceStop > 1.75f)
                    {
                        transform.Rotate(Vector3.up, Time.deltaTime * 55f);//Rotate the Car
                    }
                }
                break;
            case carBehavior.Complete:

                break;
            default:
                break;
        }
    }
}
