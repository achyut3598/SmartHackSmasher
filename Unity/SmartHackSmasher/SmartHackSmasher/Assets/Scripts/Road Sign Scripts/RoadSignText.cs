using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class RoadSignText : MonoBehaviour
{
    public TextMeshProUGUI hackedText, normalText, demoText;
    public GameObject sixty, eighty,sixtyOne,eightyOne;
    private float counter = 0f;
    // Start is called before the first frame update
    void Start()
    {
        sixty.SetActive(false);
        eighty.SetActive(false);
        sixtyOne.SetActive(false);
        eightyOne.SetActive(false);
        normalText.text = "";
        hackedText.text = "";
        demoText.text = "Welcome to the second sign obstruction Demonstration\n\nIn This Demonstration, both cars will attempt to get on the highway";
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

    public void changeToSignState()
    {
        normalText.text = "";
        hackedText.text = "";
        //Maybe play a hackery sound effect here
        demoText.text = "Notice the Sign is Obscured for both cars";

    }
    public void changeToSignExplainState()
    {
        normalText.text = "SmartHackSmasher has 2 Modules that Detect 60 MPH and 1 That detects 80MPH.  The voting system then decides on 60MPH";
        hackedText.text = "Without SmartHackSmasher, the vehicle goes with what its single system detects, 80MPH";
        //Maybe play a hackery sound effect here
        demoText.text = "";
        sixty.SetActive(true);
        eighty.SetActive(true);
        sixtyOne.SetActive(true);
        eightyOne.SetActive(true);
    }
    public void changeToSpeedUpState()
    {
        normalText.text = "SmartHackSmasher Enabled";
        hackedText.text = "SmartHackSmasher Disabled";
        demoText.text = "";
        sixtyOne.SetActive(false);
        eightyOne.SetActive(false);
    }

    public void changeToExplainState()
    {
        sixty.SetActive(false);
        eighty.SetActive(false);
        normalText.text = "SmartHackSmasher's algorithms voted on the speed and ended up being safer";
        hackedText.text = "The regular car only has one AI running the show and gets some air.";
        demoText.text = "";
    }
} 
