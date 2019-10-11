<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Viberr</title>
    
    <link rel="shortcut icon" type="image/png" href="/static/favicon.ico"/>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css">
    <link href='https://fonts.googleapis.com/css?family=Satisfy' rel='stylesheet' type='text/css'>
    <link rel="stylesheet" type="text/css" href="/static/music/style.css"/>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.0/jquery.min.js"></script>
    <script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js"></script>

    <script src="/static/music/js/main.js"></script>
    <script >
    function run() {
        
        var str=document.forms["reg"]["numberplate"].value;
        window.location="new.php?var="+str;
     
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
<nav class="navbar navbar-inverse">
    <div class="container-fluid">

        <!-- Header -->
        <div class="navbar-header">
            <button type="button" class="navbar-toggle" data-toggle="collapse" data-target="#topNavBar">
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
            </button>
            <a class="navbar-brand" href="/music/">BLACK DRAGONS</a>
        </div>

        <!-- Items -->
        <div class="collapse navbar-collapse" id="topNavBar">
            <ul class="nav navbar-nav">
                <li class="active"><a href="/music/"><span class="glyphicon glyphicon-user" aria-hidden="true"></span>&nbsp; Travellers</a></li>
                <li class=""><a href="/music/songs/all/"><span class="glyphicon glyphicon-screenshot" aria-hidden="true"></span>&nbsp; Track</a></li>
            </ul>
            
            <ul class="nav navbar-nav navbar-right">
                <form class="navbar-form navbar-left" role="search" name="reg" method="POST" >
                <div class="form-group">
                    <input type="text" class="form-control" id="numberplate"name="numberplate" value="">
                </div>
                <button  type="button" class="btn btn-default" onclick="run()">Search</button>
            </form>
                <li>
                    <a href="/music/logout_user/">
                        <span class="glyphicon glyphicon-off" aria-hidden="true"></span>&nbsp; Logout
                    </a>
                </li>
            </ul>
        </div>

    </div>
</nav>

<div class="albums-container container-fluid">

    <!-- Albums -->
    <div class="row">
        <div class="col-sm-12">
            <h3>MR.MECHANICAL's Albums</h3>
        </div>
        
            
                <div class="col-sm-4 col-lg-2">
                    <div class="thumbnail">
                        <a href="/music/1/">
                            <img src="/media/2.png" class="img-responsive">
                        </a>
                        <div class="caption">
                            <h2>as</h2>
                            <h4>as</h4>

                            <!-- View Details -->
                            <a href="/music/1/" class="btn btn-primary btn-sm" role="button">View Details</a>

                            <!-- Delete Album -->
                            <form action="/music/1/delete_album/" method="post" style="display: inline;">
                                <input type="hidden" name="csrfmiddlewaretoken" value="Bcs1Fah0sD8EayfHGBhmxeh0YSYwdhm6tzZ62UIPkZ7ZTJkAwEoy9kiqbd5XlxT9">
                                <input type="hidden" name="album_id" value="1" />
                                <button type="submit" class="btn btn-default btn-sm">
                                    <span class="glyphicon glyphicon-trash"></span>
                                </button>
                            </form>

                            <!-- Favorite Album -->
                            <a href="/music/1/favorite_album/" class="btn btn-default btn-sm btn-favorite" role="button">
                                <span class="glyphicon glyphicon-star "></span>
                            </a>

                        </div>
                    </div>
                </div>
                
            
                <div class="col-sm-4 col-lg-2">
                    <div class="thumbnail">
                        <a href="/music/2/">
                            <img src="/media/learn.kalilinux.tutorial.png" class="img-responsive">
                        </a>
                        <div class="caption">
                            <h2>mecj</h2>
                            <h4>mech</h4>

                            <!-- View Details -->
                            <a href="/music/2/" class="btn btn-primary btn-sm" role="button">View Details</a>

                            <!-- Delete Album -->
                            <form action="/music/2/delete_album/" method="post" style="display: inline;">
                                <input type="hidden" name="csrfmiddlewaretoken" value="Bcs1Fah0sD8EayfHGBhmxeh0YSYwdhm6tzZ62UIPkZ7ZTJkAwEoy9kiqbd5XlxT9">
                                <input type="hidden" name="album_id" value="2" />
                                <button type="submit" class="btn btn-default btn-sm">
                                    <span class="glyphicon glyphicon-trash"></span>
                                </button>
                            </form>

                            <!-- Favorite Album -->
                            <a href="/music/2/favorite_album/" class="btn btn-default btn-sm btn-favorite" role="button">
                                <span class="glyphicon glyphicon-star "></span>
                            </a>

                        </div>
                    </div>
                </div>
                
            
        
    </div>

    <!-- If user searches and there are songs -->
    

</div>

</body>
</html>
