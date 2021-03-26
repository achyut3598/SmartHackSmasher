
function prompt_alerts(description) {
  alert(description);
}

function readURL(input) {
  if (input.files && input.files[0]) {
      var reader = new FileReader();
      reader.onload = function (e) {
          $('#imageResult')
              .attr('src', e.target.result);
      };
      reader.readAsDataURL(input.files[0]);
  }
}

var input = document.getElementById('customFile-1');

$(function () {
  $('#customFile-1').on('change', function () {
      readURL(input);
  });
});

$("#customFile-1").change(function(){
  $("#file-name-1").text(this.files[0].name);
      readURL(input);
});

$("#customFile-2").change(function(){
  $("#file-name-2").text(this.files[0].name);
});

$("#customFile-3").change(function(){
  $("#file-name-3").text(this.files[0].name);
  gif_display();
});

function myNewFunction(){
  eel.expose(prompt_alerts);
  eel.get_random_number();
}

async function pick_file() {
  let file_loc = document.getElementById('customFile-1').files[0].name;
  let curr = document.getElementById('sel1').value;
  var pos = curr.indexOf(":");
  curr = curr.substring(0,pos);
  let file_div = document.getElementById('output-display');
  
  // Call into Python so we can access the file system
  let output = await eel.road_main(file_loc, curr)();
  if(output == "No Anomaly Detected"){
    file_div.style.color = "green";
  } else{
    file_div.style.color = "red";
  }
  file_div.innerHTML = output;

}

async function gps_pick_file() {
  let file_loc = document.getElementById('customFile-2').files[0].name;
  let curr = document.getElementById('stringtext-1').value;
  let dest = document.getElementById('stringtext-2').value;
  let file_div = document.getElementById('output-display-1');
  
  // Call into Python so we can access the file system
  let output = await eel.gps_main(file_loc, curr, dest)();
  if(output == "No Anomaly Detected"){
    file_div.style.color = "green";
    file_div.innerHTML = output;
  } else{
    file_div.style.color = "red";
    file_div.innerHTML = output + '<br />'+ 'Car is no longer headed to the destination';
  }
}

async function gif_display() {
  let file_loc = document.getElementById('customFile-3').files[0].name;
  let output = await eel.gif_maker(file_loc)();
  document.getElementById("imageResult-1").src = 'out.gif';
}

async function object_pick_file() {
  let file_loc = document.getElementById('customFile-3').files[0].name;
  let file_div = document.getElementById('output-display-2');
  // Call into Python so we can access the file system
  let output = await eel.obj_main(file_loc)();
  if(output == "Anomaly Detected"){
    file_div.style.color = "red";
  } else{
    file_div.style.color = "green";
  }
  file_div.innerHTML = output;
}
 
async function combined_pick_file() {
  let file_loc_road = document.getElementById('customFile-1').files[0].name;
  let file_loc_gps = document.getElementById('customFile-2').files[0].name;
  let file_loc_obj = document.getElementById('customFile-3').files[0].name;
  let file_div = document.getElementById('output-display-3');
  let curr = document.getElementById('sel1').value;
  let source = document.getElementById('stringtext-1').value;
  let dest = document.getElementById('stringtext-2').value;
  var pos = curr.indexOf(":");
  curr = curr.substring(0,pos);
  let output_road = await eel.road_main(file_loc_road, curr)();
  let output_gps = await eel.gps_main(file_loc_gps, source, dest)();
  let output_obj = await eel.obj_main(file_loc_obj)();
  file_div.innerHTML = output_road + '<br />';
  file_div.innerHTML += output_gps + '<br />';
  file_div.innerHTML += output_obj;
}
