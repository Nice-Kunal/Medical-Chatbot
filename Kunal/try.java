public class try {
    public static void main(String[] args) {
        int = 10,b=0,c;
        try{
            
            c = a/b;
            System.out.println(c);
        }
catch (ArithmeticException e) {
    System.out.println("Divide by zero" + e.getMessage());
}
    }
}
