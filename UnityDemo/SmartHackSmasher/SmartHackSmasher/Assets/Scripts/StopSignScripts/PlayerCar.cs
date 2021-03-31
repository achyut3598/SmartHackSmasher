using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerCar : MonoBehaviour
{
    public float speed = 20.0f;
    public bool isHackedCar = false;
    public float timeSinceStop = 0f;
    private Rigidbody rigidbody;

    public enum carBehavior { Straight, Slow, Stop, Turn, Crash, Complete };
    public carBehavior currentCarBehavior;


    // Start is called before the first frame update
    void Start()
    {
        currentCarBehavior = carBehavior.Straight;
        rigidbody = GetComponent<Rigidbody>();
    }

    void Update()
    {
        switch (currentCarBehavior)
        {
            case carBehavior.Straight:
                transform.Translate(Vector3.forward * Time.deltaTime * speed);//Moves Forward based on Verticl Input
                break;
            case carBehavior.Slow:
                if (Time.timeScale >0)
                {
                    speed *= .99f;
                }
                transform.Translate(Vector3.forward * Time.deltaTime * speed);//Moves Forward
                break;
            case carBehavior.Stop:
                timeSinceStop += (Time.deltaTime * Time.timeScale);
                if (isHackedCar || timeSinceStop > 1.5f)
                {
                    timeSinceStop = 0;
                    currentCarBehavior = carBehavior.Turn;
                }

                break;
            case carBehavior.Turn:
                timeSinceStop += (Time.deltaTime * Time.timeScale);
                transform.Translate(Vector3.forward * Time.deltaTime * 4f);//Moves Forward 
                if (timeSinceStop > .5f)
                {
                    transform.Rotate(Vector3.up, Time.deltaTime * 45f);//Rotate the Car
                }
                if (timeSinceStop > 2.5f)
                {
                    speed = 10;
                    currentCarBehavior = carBehavior.Straight;
                }
                break;
            case carBehavior.Crash:
                timeSinceStop += (Time.deltaTime * Time.timeScale);
                if (!isHackedCar)
                {
                    transform.Translate(Vector3.forward * Time.deltaTime * speed);//Moves Forward based on Verticl Input
                }
                else
                {
                    transform.Rotate(Vector3.right * Time.timeScale* 12f);//Rotate the Car
                    transform.Rotate(Vector3.up * Time.timeScale * 12f);//Rotate the Car
                    transform.Translate(Vector3.right * Time.deltaTime  * 20f);//Moves Forward based on Verticl Input
                    transform.Translate(Vector3.up * Time.deltaTime * .1f);//Moves Forward based on Verticl Input
                }
                if (timeSinceStop > 3f)
                {
                    currentCarBehavior = carBehavior.Complete;
                }
                break;
            case carBehavior.Complete:
                transform.Translate(Vector3.forward * 0);
                transform.Rotate(Vector3.up * 0);
                rigidbody.velocity = new Vector3(0, 0, 0);
                break;
            default:
                break;
        }
    }
}
