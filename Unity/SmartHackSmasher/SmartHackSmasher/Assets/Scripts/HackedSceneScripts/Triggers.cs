using Assets.Scripts.HackedSceneScripts;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Triggers : MonoBehaviour
{
    // Start is called before the first frame update
    public enum typeOfTrigger { Slow, Stop, Park, Explain, Crash };
    public typeOfTrigger type;
    public GameObject demoObject;
    private DemonstrationScript demoScript;
    void Start()
    {
        demoScript = demoObject.GetComponent<DemonstrationScript>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    private void OnTriggerEnter(Collider other)
    {
        RegularCarController car = other.gameObject.GetComponent<RegularCarController>();


        if (car != null)
        {
            switch (type)
            {
                case typeOfTrigger.Slow:
                    car.currentCarBehavior = RegularCarController.carBehavior.Slow;
                    demoScript.currentState = DemonstrationScript.State.Slowing;
                    break;
                case typeOfTrigger.Stop:
                    car.currentCarBehavior = RegularCarController.carBehavior.Stop;
                    demoScript.currentState = DemonstrationScript.State.Hacking;
                    demoScript.counter = 0f;
                    break;
                case typeOfTrigger.Park:
                    car.currentCarBehavior = RegularCarController.carBehavior.Park;
                    demoScript.currentState = DemonstrationScript.State.Parking;
                    demoScript.counter = 0f;
                    break;
                case typeOfTrigger.Explain:
                    demoScript.currentState = DemonstrationScript.State.Explaining;
                    demoScript.counter = 0f;
                    break;
                case typeOfTrigger.Crash:
                    car.currentCarBehavior = RegularCarController.carBehavior.Complete;
                    demoScript.counter = 0f;
                    demoScript.currentState = DemonstrationScript.State.Crashing;
                    break;
            }
        }
    }
}
