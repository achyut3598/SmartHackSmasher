using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RoadSignColision : MonoBehaviour
{
    // Start is called before the first frame update
    public enum typeOfTrigger { Sign, SpeedUp, OnRamp,Explain};
    public typeOfTrigger type;
    public GameObject demoObject;
    private RoadSignDemo demoScript;
    private bool hasColided = false;
    void Start()
    {
        demoScript = demoObject.GetComponent<RoadSignDemo>();
    }

    // Update is called once per frame
    void Update()
    {

    }
    private void OnTriggerEnter(Collider other)
    {
        RoadSignPlayer car = other.gameObject.GetComponent<RoadSignPlayer>();


        if (car != null&&!hasColided)
        {
            hasColided = true;
            switch (type)
            {
                case typeOfTrigger.Sign:
                    demoScript.currentState = RoadSignDemo.State.RoadSign;
                    break;
                case typeOfTrigger.SpeedUp:
                    car.currentCarBehavior = RoadSignPlayer.carBehavior.SpeedUp;
                    demoScript.currentState = RoadSignDemo.State.SpeedingUp;
                    demoScript.counter = 0f;
                    break;
                case typeOfTrigger.OnRamp:
                    car.currentCarBehavior = RoadSignPlayer.carBehavior.OnRamp;
                    demoScript.currentState = RoadSignDemo.State.OnRamp;
                    demoScript.counter = 0f;
                    break;
                case typeOfTrigger.Explain:
                    car.timeSinceStop = 0f;
                    car.currentCarBehavior = RoadSignPlayer.carBehavior.Complete;
                    demoScript.counter = 0f;
                    demoScript.currentState = RoadSignDemo.State.Explaining;
                    break;

            }
        }
    }
}
