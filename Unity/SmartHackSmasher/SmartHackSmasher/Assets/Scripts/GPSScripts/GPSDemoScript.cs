using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GPSDemoScript : MonoBehaviour
{
    public enum State { Initial, Driving, Hacking1, Hacking2, Explaining, Done };
    public State currentState;
    public GameObject textObjects;
    public float counter = 0f;


    private GPSTextScript textScript;



    // Start is called before the first frame update
    void Start()
    {
        currentState = State.Initial;
        textScript = textObjects.GetComponent<GPSTextScript>();
        PauseGame();
    }

    // Update is called once per frame
    void Update()
    {
        counter += 1;
        switch (currentState)
        {
            case State.Initial:
                textScript.changeDirectionText("Turn Right in 1000 Feet", "Turn Right in 1000 Feet");
                PauseGame();
                if (counter >= 300f)
                {
                    currentState = State.Driving;
                }
                break;
            case State.Driving:
                ResumeGame();
                textScript.changeToDrivingState();
                counter = 0f;
                break;
            case State.Hacking1:
                textScript.changeDirectionText("Turn Right in 500 Feet", "Turn Right in 500 Feet");
                textScript.changeToHackingState1();
                PauseGame();
                if (counter > 300f)
                {
                    ResumeGame();
                    counter = 0f;
                    currentState = State.Driving;
                }
                break;
            case State.Hacking2:
                textScript.xOffset = -300f;
                textScript.changeDirectionText("Turn Right in 1500 Feet", "Turn Right in 1500 Feet");
                textScript.changeToHackingState2();
                PauseGame();
                if (counter > 300f)
                {
                    textScript.changeToHackingState3();
                    textScript.changeDirectionText("Turn Right in 500 Feet", "Turn Right in 1500 Feet");
                }
                if (counter > 800f)
                {
                    ResumeGame();
                    counter = 0f;
                    currentState = State.Driving;
                }
                break;
            case State.Explaining:
                PauseGame();
                textScript.changeToExplainState();
                break;
            case State.Done:
                break;
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

    public void updateGPSDirections(string normalText, string hackedText)
    {
        textScript.changeDirectionText(normalText, hackedText);
    }
    public void updateNormalGPSDirections(string normalText)
    {
        textScript.changeNormalDirectionText(normalText);
    }
    public void updateHackedGPSDirections(string hackedText)
    {
        textScript.changeHackedDirectionText(hackedText);
    }
}
