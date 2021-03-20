using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class StopSignColision : MonoBehaviour
{
    // Start is called before the first frame update
    public enum typeOfTrigger { Slow, Stop, Turn, Crash,Explain};
    public typeOfTrigger type;
    public GameObject demoObject;
    private StopSignDemonstration demoScript;
    private bool hasColided = false;
    void Start()
    {
        demoScript = demoObject.GetComponent<StopSignDemonstration>();
    }

    // Update is called once per frame
    void Update()
    {

    }
    private void OnTriggerEnter(Collider other)
    {
        PlayerCar car = other.gameObject.GetComponent<PlayerCar>();


        if (car != null&&!hasColided)
        {
            hasColided = true;
            switch (type)
            {
                case typeOfTrigger.Slow:
                    car.currentCarBehavior = PlayerCar.carBehavior.Slow;
                    demoScript.currentState = StopSignDemonstration.State.Slowing;
                    break;
                case typeOfTrigger.Stop:
                    car.currentCarBehavior = PlayerCar.carBehavior.Stop;
                    demoScript.currentState = StopSignDemonstration.State.Stopping;
                    demoScript.counter = 0f;
                    break;
                case typeOfTrigger.Turn:
                    car.currentCarBehavior = PlayerCar.carBehavior.Turn;
                    demoScript.currentState = StopSignDemonstration.State.Crashing;
                    demoScript.counter = 0f;
                    break;
                case typeOfTrigger.Crash:
                    car.timeSinceStop = 0f;
                    car.currentCarBehavior = PlayerCar.carBehavior.Crash;
                    demoScript.counter = 0f;
                    demoScript.currentState = StopSignDemonstration.State.Crashing;
                    break;
                case typeOfTrigger.Explain:
                    car.currentCarBehavior = PlayerCar.carBehavior.Complete;
                    demoScript.currentState = StopSignDemonstration.State.Explaining;
                    demoScript.counter = 0f;
                    break;

            }
        }
    }
}
