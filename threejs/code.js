function name1(data){
	// console.log(data);

	// var positions = [[1., 3., -1.],[ 1., -1., -1.],[5., 1., -1.]];
	var positions = data;


	var scene = new THREE.Scene();

	var WIDTH = 400, HEIGHT = 300;
	var VIEW_ANGLE = 100;
	var ASPECT = window.innerWidth/window.innerHeight;
	var NEAR = 0.1;
	var FAR = 1000;
	var camera = new THREE.PerspectiveCamera(VIEW_ANGLE, ASPECT, NEAR, FAR);

	var renderer = new THREE.WebGLRenderer();
	renderer.setSize( window.innerWidth, window.innerHeight );
	document.body.appendChild( renderer.domElement );


	// get min dist between any nodes
	min_dist = -1;
	for (i = 0; i < positions.length; i++)
	{ 
		for (j = i+1; j < positions.length; j++)
		{
			dist = 0;
			for (k = 0; k < 3; k++)
			{
				dist = dist + Math.pow(positions[i][k] - positions[j][k], 2);
			}
			dist = Math.sqrt(dist)

			if (dist < min_dist || min_dist == -1)
			{
				min_dist = dist
			}
		
			// console.log(dist);
		}
	}
	// min_dist = min_dist
	min_dist = min_dist /2
	console.log(min_dist);

	var material = new THREE.MeshBasicMaterial( { color: 0x00ff00, transparent: true, opacity:.5, } );
	var geometry = new THREE.SphereGeometry(min_dist,16,16)

	for (i = 0; i < positions.length; i++)
	{ 
		var sphere = new THREE.Mesh(geometry,material);
		sphere.position.x = positions[i][0];
		sphere.position.y = positions[i][1];
		sphere.position.z = positions[i][2];

		scene.add(sphere);
	}

	var material = new THREE.LineBasicMaterial({color: 0x0000ff});

	for (i = 0; i < positions.length-1; i++)
	{ 
		var geometry = new THREE.Geometry();
		geometry.vertices.push(new THREE.Vector3(positions[i][0], positions[i][1], positions[i][2]),new THREE.Vector3(positions[i+1][0], positions[i+1][1], positions[i+1][2]));
		var line = new THREE.Line( geometry, material );
		scene.add( line );
	}


	camera.position.z = 5;
	camera.lookAt(geometry);
	// camera.lookAt(scene.position);
	// camera.position.set(0,150,400)
	renderer.setClearColor( 0xdddddd, 1);

	var render = function () {
		requestAnimationFrame( render );

		scene.rotation.y += 0.001;
		scene.rotation.x += 0.001;
		scene.rotation.z += 0.000;

		renderer.render(scene, camera);
	};

	render();
}

$.ajax({
	dataType: "json",
	url: '/predicted_RASH2.json',
	success: name1,
	error: function(ding) { console.log('error')}
});


