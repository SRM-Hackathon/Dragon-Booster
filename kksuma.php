<!DOCTYPE html>
<html>
<head>
    
<style>
table {
    width: 100%;
    border-collapse: collapse;
}

table, td, th {
    border: 1px solid black;
    padding: 5px;
}

th {text-align: left;}
</style>
<script>
    var div = document.getElementById("dom-target");
    var myData = div.textContent;

    function run() {
        
        var str=document.forms["reg"]["numberplate"].value;
        window.location="map.php?var="+myData;
     
}

$(document).ready(function(){
$("input").click(function(){
        $(this).next().show();
        $(this).next().hide();
    });

});

</script>

</head>
<body>

<div id="dom-target" ">
    <?php 
      


$con = mysqli_connect('localhost','root','','placemain');
if (!$con) {
    die('Could not connect: ' . mysqli_error($con));
}

//mysqli_select_db($con,"ajax_demo");
$sql="SELECT * FROM details ";
$result = mysqli_query($con,$sql);

echo "<table>";
echo "<tr>";
echo "<th> FIRST NAME </th> <th>NUMBER PLATE </th><th>Vehicle_type </th><th>Place </th> <th>Time  Stamp</th>";
echo"</tr>";
while($row = mysqli_fetch_array($result)) {

    echo "<tr>";
    echo "<th>" . $row['First_NAME'] . "</th> <th>". $row['No_Plate'] ."</th> <th>". $row['Vehicle_type'] ."</th> <th>". $row['Location'] ."</th> <th>". $row['Time_stamp'] ."</th>";
    echo "</tr>";
   $output = $row['First_NAME'];
}
echo "</table>";
  
mysqli_close($con);
?>
</div>
</body>
</html>