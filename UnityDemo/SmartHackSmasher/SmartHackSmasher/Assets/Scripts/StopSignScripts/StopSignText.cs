using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class StopSignText : MonoBehaviour
{
    public TextMeshProUGUI hackedText, normalText, demoText;

    // Start is called before the first frame update
    void Start()
    {
        normalText.text = "";
        hackedText.text = "";
        demoText.text = "Welcome to the first sign obstruction Demonstration\n\nIn This Demonstration, both cars will attempt to turn onto a busy road";
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

    public void changeToSlowingState()
    {
        normalText.text = "";
        hackedText.text = "";
        //Maybe play a hackery sound effect here
        demoText.text = "Notice the Stop Sign Sign is Obscured for both cars";
    }

    public void changeToTurningState()
    {
        normalText.text = "SmartHackSmasher's algorithms are able to take a reasonable guess that there is a stop sign behind the tree";
        hackedText.text = "The regular car fails to see the obscured sign so turns right away.";
        demoText.text = "";
    }
    public void changeToCrashState()
    {
        normalText.text = "";
        hackedText.text = "";
        demoText.text = "Not recognizing that there was a stop sign, the car without SmartHackSmasher gets hit by a car in traffic";
    }
    public void changeToExplanationState()
    {
        normalText.text = "";
        hackedText.text = "";
        demoText.text = "The car without SmartHackSmasher failed to notice the obscured sign and crashed.\n\nThe car with SmartHackSmasher was able to detect that this looked like a place where a stop sign should be";
    }
}
