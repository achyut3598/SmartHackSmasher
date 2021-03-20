using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GPSPlayerScript : MonoBehaviour
{
    public float speed = 10.0f;
    public bool isHackedCar = false;
    public float timeSinceStop = 0f;
    private float turnSpeed = 10f;
    private Rigidbody rigidbody;
    private float angleTurned = 0f;
    private float dirToTurn = -90f;
    public enum carBehavior { Straight, TurnAlongRoad, StraightenOut, RightTurn, TurnIntoLot, Complete };
    public carBehavior currentCarBehavior;

    // Start is called before the first frame update
    void Start()
    {
        currentCarBehavior = carBehavior.Straight;
        rigidbody = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        switch (currentCarBehavior)
        {
            case carBehavior.Straight:
                transform.Translate(Vector3.forward * Time.deltaTime * speed);//Moves Forward based on Verticl Input
                break;
            case carBehavior.TurnAlongRoad:
                angleTurned += turnSpeed;
                transform.Rotate(Vector3.up, Time.deltaTime * turnSpeed);//Rotate the Car
                transform.Translate(Vector3.forward * Time.deltaTime * speed);//Moves Forward based on Verticl Input
                break;
            case carBehavior.StraightenOut:
                transform.eulerAngles = new Vector3(0, dirToTurn, 0);
                currentCarBehavior = carBehavior.Straight;
                break;
            case carBehavior.RightTurn:
                dirToTurn = 90f;
                timeSinceStop += (Time.deltaTime * Time.timeScale);
                transform.Translate(Vector3.forward * Time.deltaTime * 4f);//Moves Forward 
                transform.Rotate(Vector3.up, Time.deltaTime * 45f);//Rotate the Car
                if (timeSinceStop > 2f)
                {
                    currentCarBehavior = carBehavior.Straight;
                }
                break;
            case carBehavior.TurnIntoLot:
                angleTurned = 0f;
                transform.Rotate(Vector3.up, Time.deltaTime * turnSpeed*2.5f);//Rotate the Car
                transform.Translate(Vector3.forward * Time.deltaTime * speed);//Moves Forward based on Verticl Input
                break;
            case carBehavior.Complete:
                transform.Translate(Vector3.forward * 0);
                rigidbody.velocity = new Vector3(0, 0, 0);
                break;
            default:
                break;
        }
    }
}
