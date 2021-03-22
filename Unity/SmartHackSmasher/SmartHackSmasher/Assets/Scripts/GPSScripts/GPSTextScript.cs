using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class GPSTextScript : MonoBehaviour
{
    public TextMeshProUGUI hackedText, normalText, demoText,hackedGPSText,normalGPSText,hackedDirectionText,normalDirectionText;
    public GameObject playerCarObject,hackedCarObject,redObject,greenObject;

    public float xOffset = 0f;
    private bool stopDisplayingBoxes = false;

    // Start is called before the first frame update
    void Start()
    {
        normalText.text = "";
        hackedText.text = "";
        demoText.text = "Welcome to the GPS Demonstration\n\nIn This Demonstration, a hacker will feed false GPS Coordinates into a car's navigation system.";
        redObject.SetActive(false);
        greenObject.SetActive(false);
    }

    // Update is called once per frame
    void Update()
    {
        float x = playerCarObject.transform.position.x;
        float y = playerCarObject.transform.position.z;
        normalGPSText.text = "Lat:" + (x - xOffset) + "\nLon:" + y;
        x = hackedCarObject.transform.position.x;
        y = hackedCarObject.transform.position.z;
        hackedGPSText.text = "Lat:" + (x - xOffset) + "\nLon:" + y;
    }

    public void changeToDrivingState()
    {
        normalText.text = "SmartHackSmasher Enabled";
        hackedText.text = "SmartHackSmasher Disabled";
        demoText.text = "";
        if (stopDisplayingBoxes)
        {
            redObject.SetActive(false);
            greenObject.SetActive(false);

        }
    }

    public void changeToHackingState1()
    {
        normalText.text = "";
        hackedText.text = "";
        //Maybe play a hackery sound effect here
        demoText.text = "Pay Attention to the cooridinates as the van attempts to hack the car";
        redObject.SetActive(true);
        greenObject.SetActive(true);
    }

    public void changeToHackingState2()
    {
        normalText.text = "";
        hackedText.text = "";
        //Maybe play a hackery sound effect here
        demoText.text = "Notice that both cars cooridinates have jumped back significantly on the road";
        stopDisplayingBoxes = true;

    }

    public void changeToHackingState3()
    {
        normalText.text = "";
        hackedText.text = "";
        //Maybe play a hackery sound effect here
        demoText.text = "SmartHackSmasher detects the GPS is throwing a value that doesn't match up with the rest of its sesnors and chooses to enter a fallback mode.\n\nThe normal car takes the new gps coordinates as truth.";
    }


    public void changeToExplainState()
    {
        normalText.text = "SmartHackSmasher's algorithms are able to detect the GPS value randomly jumping which does not match up with other sensors like velocity";
        hackedText.text = "The regular car reads the coordinates as is and finds itself in a sticky situation";
        demoText.text = "";
    }
    public void changeDirectionText(string normalText, string hackedText)
    {
        normalDirectionText.text = normalText;
        hackedDirectionText.text = hackedText;
    }

    public void changeNormalDirectionText(string text)
    {
        normalDirectionText.text = text;
    }

    public void changeHackedDirectionText(string text)
    {
        hackedDirectionText.text = text;
    }
}
