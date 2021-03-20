using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Assets.Scripts.HackedSceneScripts
{
    class RegularCarController : MonoBehaviour
    {
        public float speed = 20.0f;
        public bool isHackedCar = false;
        private float timeSinceStop = 0f;
        private float parkingCounter = 0f;
        private Rigidbody rigidbody;

        public enum carBehavior { Straight, Slow, Stop, Turn, Park, Complete };
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
                    speed *= .99f;
                    transform.Translate(Vector3.forward * Time.deltaTime * speed);//Moves Forward
                    break;
                case carBehavior.Stop:
                    timeSinceStop += (Time.deltaTime * Time.timeScale);
                    if(timeSinceStop > 1.5f)
                    {
                        timeSinceStop = 0;
                        currentCarBehavior = carBehavior.Turn;
                    }
                    break; 
                case carBehavior.Turn:
                    timeSinceStop += (Time.deltaTime* Time.timeScale);
                    transform.Translate(Vector3.forward * Time.deltaTime * 4f);//Moves Forward 
                    if (timeSinceStop > 1.5f)
                    {
                        transform.Rotate(Vector3.down, Time.deltaTime * 45f);//Rotate the Car
                    }
                    if (timeSinceStop > 3.47f)
                    {
                        speed = 10;
                        currentCarBehavior = carBehavior.Straight;
                    }
                    break;
                case carBehavior.Park:
                    parkingCounter += (Time.deltaTime * Time.timeScale);
                    if(parkingCounter < 3f)
                    {
                        transform.Translate(Vector3.forward * Time.deltaTime * 2f);//Moves Forward 
                    }
                    else if (parkingCounter < 5.25f && isHackedCar)
                    {
                        transform.Rotate(Vector3.up, Time.deltaTime * 10f);//Rotate the Car
                        transform.Translate(Vector3.forward * Time.deltaTime * 2f);//Moves Forward 
                    }
                    else if (parkingCounter < 6f && isHackedCar)
                    {
                        transform.Translate(Vector3.forward * Time.deltaTime * 4f);//Moves Forward 
                    }
                    else if (parkingCounter < 8f && isHackedCar)
                    {
                        transform.Rotate(Vector3.down, Time.deltaTime * 10f);//Rotate the Car
                    }
                    else if (parkingCounter < 9.35f&& isHackedCar)
                    {
                        transform.Translate(Vector3.forward * Time.deltaTime * 2f);//Moves Forward 
                        transform.Translate(Vector3.left * Time.deltaTime * .5f);//Moves Forward 
                    }
                    else
                    {
                        currentCarBehavior = carBehavior.Complete;
                    }
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
}
