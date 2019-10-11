
<!DOCTYPE html>
<html>
<head>

    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">

    <link rel="stylesheet" type="text/css" href="main.css">

	
		<title>Login</title>
	
</head>
<?php
$emerr="message to check";













?>


<script>
	function check()
	{
	
		var mail=document.forms["reg"]["email"].value;
		var pass=document.forms["reg"]["password"].value;
	
		
		


		if(mail=="") 
		{
			window.alert("ENTER EMAIL");
		}
		else
		{
			var c6=/^[\w]+([\.-]?\w+)*@\w+([\.]?\w)*([\.]?\w{2,3})+$/g;

			if(!c6.test(mail))
				window.alert("ENTER VALID EMAIL-ID");
		}

		if(pass=="") 
		{
			window.alert("ENTER PASSWORD");
		}
		else
		{
			var c2=/[\w\W]{2,20}/g;
			if(!c2.test(pass))
				window.alert("ENTER VALID PASSWORD");
		}
		
		
	}
	</script>



<body>

	<header class="site-header">
	  <nav class="navbar navbar-expand-md navbar-dark bg-dark fixed-top sidebarNavigation" data-sidebarClass="navbar-dark bg-dark">
	    <div class="container">
	      <a class="navbar-brand mr-4" href="home.php">BLACK DRAGONS</a>
	      <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarToggle" aria-controls="navbarToggle" aria-expanded="false" aria-label="Toggle navigation">
	        <span class="navbar-toggler-icon"></span>
	      </button>
	      <div class="collapse navbar-collapse" id="navbarToggle">
	        <div class="navbar-nav mr-auto">
	          <a class="nav-item nav-link" href="home.php">Home</a>
	          <a class="nav-item nav-link" href="about.php">About</a>
	        </div>
	        <!-- Navbar Right Side -->
	        <div class="navbar-nav">
	          <a class="nav-item nav-link" href="login.php">Login</a>
	          <a class="nav-item nav-link" href="register.php">Register</a>
	        </div>
	      </div>
	    </div>
	  </nav>

	</header>

	<main role="main" class="container">
	  <div class="row">
	    <div class="col-md-8">
	    	
	    			
	    			
	      
	<div class="content-section">
		<form method="POST" name="reg" action="">
			
			<fieldset class="form-group">
				<legend class="border-bottom mb-4"> LOGIN PAGE</legend>
			

			<div class="form-group">
				<label class="form-control-label" for="email">Email</label>

				
                    <input class="form-control form-control-lg" id="email" name="email" required type="text" value="">
                    <span id="span"> <?php echo $emerr; ?> </span>
              	
			</div>
			<div class="form-group">
				<label class="form-control-label" for="password">Password</label>

				
                    <input class="form-control form-control-lg" id="password" name="password" required type="password" value="">
              	
			</div>
			

			</fieldset>

			<div class="form-group">
				<input class="btn btn-outline-info" id="submit" name="submit" type="submit" onclick="check()" value="Login In">
			</div>

		</form>
	</div>

	<div class="border-top pt-3">
        <small class="text-muted">
            New User? <a class="ml-2" href="register.php">Sign Up</a>
        </small>
    </div>

	    	
	   </div>
	   
	    <!--<div class="col-md-4">
	      <div class="content-section">
	        <h3>Our Sidebar</h3>
	        <p class='text-muted'>You can put any information here you'd like.
	          <ul class="list-group">
	            <li class="list-group-item list-group-item-light">Latest Posts</li>
	            <li class="list-group-item list-group-item-light">Announcements</li>
	            <li class="list-group-item list-group-item-light">Calendars</li>
	            <li class="list-group-item list-group-item-light">etc</li>
	          </ul>
	        </p>
	      </div>
	    </div>-->
	  </div>
	</main>
	<!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
</body>
</html>