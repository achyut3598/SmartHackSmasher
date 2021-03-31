using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class TextScript : MonoBehaviour
{
    public TextMeshProUGUI hackedText, normalText, demoText;

    // Start is called before the first frame update
    void Start()
    {
        normalText.text = "";
        hackedText.text = "";
        demoText.text = "Welcome to the signal Injection Demonstration\n\nIn This Demonstration, a malicious actor in a van will attempt to hack the car's visual sensors to make it appear as if there are no cars in a parking lot.";
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void changeToDrivingState()
    {
        normalText.text = "SmartHackSmasher Enabled";
        hackedText.text = "SmartHackSmasher Disabled";
        demoText.text = "";
    }

    public void changeToHackingState()
    {
        normalText.text = "";
        hackedText.text = "";
        //Maybe play a hackery sound effect here
        demoText.text = "At the stop sign, a hacker in the van attempts to take control of the car's sesnors";
    }

    public void changeToParkingState()
    {
        normalText.text = "";
        hackedText.text = "";
        demoText.text = "Hacker Succeeds in hacking car's sensors, making it appear as if the parking lot is empty when it is not.";
    }
    public void changeToExplanationState()
    {
        normalText.text = "With SmartHackSmasher, the car is able to detect an amomaly when the parking lot cars suddenly disappear.  The car alerts the driver and switches to manual mode";
        hackedText.text = "Without SmartHackSmasher, the car sees the parking lot as empty";
        demoText.text = "";
    }
    public void changeToCrashState()
    {
        normalText.text = "";
        hackedText.text = "";
        demoText.text = "The car without SmartHackSmasher crashes into another car in the parking lot as it thought the lot was empty.\n\nThe car with SmartHackSmasher was able to detect that something was wrong with the car's systems and take emergancy measures";
    }


}
