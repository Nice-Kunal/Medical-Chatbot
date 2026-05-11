<?php
$choice=1, $a=10,$b=20;
echo"1 Add,2 Sub, 3 Mul, 4 Div";
switch($choice)
{
    case 1:
        $c = a+b;
        echo "Addition", $c;
        break;
        
    case 2:
        $c = $a-$b;
        echo "Sub", $c;
        break;
    
    case 3:
        $c = $a*$b;
        echo "Mul", $c;
        break;

    case 4:
        $c = $a/$b;
        echo "Div",$c;
        break;

    default:
    echo "Out of Choice";
}
?>