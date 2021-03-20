using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class RoadSignDemonstration : MonoBehaviour
{

    private float counter =-1f;
    private int timesPaused = 0;
    public Text myText;
    public bool runCrashDialog = true;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
            counter += 1f;
            if (counter % 300 == 0)
            {
                if (Time.timeScale == 1)
                {
                if (runCrashDialog)
                {
                    if (timesPaused == 0)
                    {
                        myText.text = "The initial speed limit is 40 MPH.  The Sign is unobstructed";
                    }
                    else if (timesPaused == 1)
                    {
                        myText.text = "The cars are proceding along the road as normal and expected";
                    }
                    else if (timesPaused == 2)
                    {
                        myText.text = "The new speed limit is 60 MPH but the sign is obstructed";
                    }
                    else if (timesPaused == 3)
                    {
                        myText.text = "The car crashes since it assumed the sign said 80MPH";
                    }
 
                }
                else
                {
                    if (timesPaused == 0)
                    {
                        myText.text = "The initial speed limit is 40 MPH.  The Sign is unobstructed";
                    }
                    else if (timesPaused == 1)
                    {
                        myText.text = "The cars are proceding along the road as normal and expected";
                    }
                    else if (timesPaused == 2)
                    {
                        myText.text = "The new speed limit is 60 MPH but the sign is obstructed";
                    }
                    else if (timesPaused == 3)
                    {
                        myText.text = "The car procedes fine since SmartHackSmasher detected an amamoly reading the sign";
                    }
                }
                timesPaused++;
                PauseGame();

            }
                else
                {
                    ResumeGame();
                    myText.text = "";
                }
            }
    }

    void PauseGame()
    {
        Time.timeScale = 0;
    }

    void ResumeGame()
    {
        Time.timeScale = 1;
    }
}
