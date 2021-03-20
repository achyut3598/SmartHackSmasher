using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GPSCollisionScript : MonoBehaviour
{
    // Start is called before the first frame update
    public enum typeOfTrigger { TurnAlongRoad, StraightenOut, Hacking1,Hacking2, RightTurn, TurnIntoLot, Stop, Explain };
    public typeOfTrigger type;
    public GameObject demoObject;
    private GPSDemoScript demoScript;
    public bool isHacked = false;
    void Start()
    {
        demoScript = demoObject.GetComponent<GPSDemoScript>();
    }

    // Update is called once per frame
    void Update()
    {

    }
    private void OnTriggerEnter(Collider other)
    {
        GPSPlayerScript car = other.gameObject.GetComponent<GPSPlayerScript>();



        if (car != null)
        {
            switch (type)
            {
                case typeOfTrigger.TurnAlongRoad:
                    car.currentCarBehavior = GPSPlayerScript.carBehavior.TurnAlongRoad;
                    break;
                case typeOfTrigger.StraightenOut:
                    car.currentCarBehavior = GPSPlayerScript.carBehavior.StraightenOut;
                    break;
                case typeOfTrigger.Hacking1:
                    if (!isHacked)
                    {
                        demoScript.currentState = GPSDemoScript.State.Hacking1;
                    } 
                    break;
                case typeOfTrigger.Hacking2:
                    if (!isHacked)
                    {
                        demoScript.currentState = GPSDemoScript.State.Hacking2;
                    }
                    break;
                case typeOfTrigger.RightTurn:
                    car.currentCarBehavior = GPSPlayerScript.carBehavior.RightTurn;
                    break;
                case typeOfTrigger.TurnIntoLot:
                    car.currentCarBehavior = GPSPlayerScript.carBehavior.TurnIntoLot;
                    break;
                case typeOfTrigger.Stop:
                    car.currentCarBehavior = GPSPlayerScript.carBehavior.Complete;
                    break;
                case typeOfTrigger.Explain:
                    if (isHacked)
                    {
                        demoScript.currentState = GPSDemoScript.State.Explaining;
                    }
                    break;
                default:
                    break;


            }
        }
    }
}
