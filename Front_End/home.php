
<!DOCTYPE html>
<html>
<head>



    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">

    <link rel="stylesheet" type="text/css" href="main.css">
    <link rel="stylesheet" type="text/css" href="css/b4_sidebar.css">
        <link rel="shortcut icon" type="image/png" href="/static/favicon.ico"/>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css">

	
		<title>Home</title>
	<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
<link rel="stylesheet" href="https://www.w3schools.com/lib/w3-theme-black.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.3.0/css/font-awesome.min.css">
<script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js"></script>

<script >
    function run() {
        
        var str=document.forms["reg"]["numberplate"].value;
        window.location="search.php?var="+str;
     
}

$(document).ready(function(){
$("input").click(function(){
        $(this).next().show();
        $(this).next().hide();
    });

});



</script>
<style>
.button {
  display: inline-block;
  border-radius: 4px;
  background-color: #343a40 ;
  border: none;
  color: #FFFFFF;
  text-align: center;
  font-size: 28px;
  padding: 20px;
  width: 200px;
  transition: all 0.5s;
  cursor: pointer;
  margin: 5px;
}

.button span {
  cursor: pointer;
  display: inline-block;
  position: relative;
  transition: 0.5s;
}

.button span:after {
  content: '\00bb';
  position: absolute;
  opacity: 0;
  top: 0;
  right: -20px;
  transition: 0.5s;
}

.button:hover span {
  padding-right: 25px;
}

.button:hover span:after {
  opacity: 1;
  right: 0;
}


footer {
    background-color: #2d2d30;
    color: #f5f5f5;
    padding: 32px;
    
  }
  footer a {
    color: #f5f5f5;
  }
  footer a:hover {
    color: #777;
    text-decoration: none;
  }  
  .form-control {
    border-radius: 0;
  }
</style>
</head>
<body style="background-image: url('maintenance-bg.jpg');" >

	<header class="site-header">
	  <nav class="navbar navbar-expand-md navbar-dark bg-dark fixed-top sidebarNavigation" data-sidebarClass="navbar-dark bg-dark">
	    <div class="container">
	      <a class="navbar-brand mr-4" href="home.php">ROAD'S EYE</a>
	      <button class="navbar-toggler leftNavbarToggler" type="button" data-toggle="collapse" data-target="#navbarToggle" aria-controls="navbarToggle" aria-expanded="false" aria-label="Toggle navigation">
	        <span class="navbar-toggler-icon"></span>
	      </button>
	      <div class="collapse navbar-collapse" id="navbarToggle">
	        <div class="navbar-nav mr-auto">
	          <a class="nav-item nav-link" href="sumatry.php"><span class="glyphicon glyphicon-user" aria-hidden="true"></span>&nbspTraveller</a>
	          <a class="nav-item nav-link" href="track.php"><span class="glyphicon glyphicon-screenshot" aria-hidden="true"></span>&nbspTrack</a>
            <a class="nav-item nav-link" href="track.php"><span class="glyphicon glyphicon-screenshot" aria-hidden="true"></span>&nbsp Monitering</a>
	        </div>
	        <!-- Navbar Right Side -->
	        <div class="navbar-nav navbar-right">
            <form class="navbar-form navbar-left" role="search" name="reg" method="POST" >
                <div class="form-group">
                    <input type="text" class="form-control" id="numberplate"name="numberplate" value="">
                </div>
                <button  type="button" class="btn btn-default" onclick="run()">Search</button>
	          
	          
	        </div>
	      </div>
	    </div>
	  </nav>

	</header>

	

<div class="albums-container container-fluid">

    <!-- Albums -->
    <div class="row">
        
        
            
                <div class="col-sm-4 col-lg-2">
                    <div class="thumbnail">
                        <a href="/music/1/">
                            <img src="a.jpg" class="img-responsive">
                        </a>
                        <div class="caption">
                            <h3>Andhra Pradesh</h3>
                           

                            <!-- View Details -->
                            <a href="/music/1/" class="btn btn-primary btn-sm" role="button">View Details</a>
                        </div>
                    </div>
                </div>

        <div class="col-sm-4 col-lg-2">
                    <div class="thumbnail">
                        <a href="/music/1/">
                            <img src="b.jpg" class="img-responsive">
                        </a>
                        <div class="caption">
                            <h2>Bihar</h2>
                            

                            <!-- View Details -->
                            <a href="/music/1/" class="btn btn-primary btn-sm" role="button">View Details</a>
                        </div>
                    </div>
                </div>

                 <div class="col-sm-4 col-lg-2">
                    <div class="thumbnail">
                        <a href="/music/1/">
                            <img src="c.jpg" class="img-responsive">
                        </a>
                        <div class="caption">
                            <h2>Chhattisgarh</h2>
                            

                            <!-- View Details -->
                            <a href="/music/1/" class="btn btn-primary btn-sm" role="button">View Details</a>
                        </div>
                    </div>
                </div>

                <div class="col-sm-4 col-lg-2">
                    <div class="thumbnail">
                        <a href="/music/1/">
                            <img src="g.jpg" class="img-responsive">
                        </a>
                        <div class="caption">
                            <h2>Gujarat</h2>
                            

                            <!-- View Details -->
                            <a href="/music/1/" class="btn btn-primary btn-sm" role="button">View Details</a>
                        </div>
                    </div>
                </div>
                <div class="col-sm-4 col-lg-2">
                    <div class="thumbnail">
                        <a href="/music/1/">
                            <img src="h.jpg" class="img-responsive">
                        </a>
                        <div class="caption">
                            <h2>Haryana</h2>
                            

                            <!-- View Details -->
                            <a href="/music/1/" class="btn btn-primary btn-sm" role="button">View Details</a>
                        </div>
                    </div>
                </div>
                <div class="col-sm-4 col-lg-2">
                    <div class="thumbnail">
                        <a href="/music/1/">
                            <img src="ka.jpg" class="img-responsive">
                        </a>
                        <div class="caption">
                            <h2>Karnataka</h2>
                            

                            <!-- View Details -->
                            <a href="/music/1/" class="btn btn-primary btn-sm" role="button">View Details</a>
                        </div>
                    </div>
                </div>
                </div>
                <div class="col-sm-4 col-lg-2">
                    <div class="thumbnail">
                        <a href="/music/1/">
                            <img src="ta.jpg" class="img-responsive">
                        </a>
                        <div class="caption">
                            <h2>Tamil Nadu</h2>
                            

                            <!-- View Details -->
                            <a href="district.php" class="btn btn-primary btn-sm" role="button">View Details</a>
                        </div>
                    </div>
                </div>
    <!--
<footer class="page-footer font-small teal pt-4">

   
    <div class="container-fluid text-center text-md-left">

      <div class="row">

       
        <div class="col-md-6 mt-md-0 mt-3">

          
          <h5 class="text-uppercase font-weight-bold">BLACK_DRAGON</h5>
        <p> ABOUT : </p>
    <p> CONTRIBUTERS : </p>
    <p> MENTOR : </p>

        </div>
      

        
     

      </div>
     

    </div>
   

    <div class="footer-copyright text-center py-3">© 2018 Copyright:
      <a href="https://mdbootstrap.com/education/bootstrap/"> BLACK_DRAGONS </a>
    </div>


  </footer> 
  -->
	<!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <script src="js/b4_sidebar.js"></script>
</body>
</html>