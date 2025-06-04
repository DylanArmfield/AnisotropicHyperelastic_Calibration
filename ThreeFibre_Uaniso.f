      subroutine uanisohyper_inv (aInv, ua, zeta, nFibers, nInv,
     *                            ui1, ui2, ui3, temp, noel,
     *                            cmname, incmpFlag, ihybFlag,
     *                            numStatev, statev,
     *                            numFieldv, fieldv, fieldvInc,
     *                            numProps, props)
C
      include 'aba_param.inc'
C
      character *80 cmname
      dimension aInv(nInv), ua(2), zeta(nFibers*(nFibers-1)/2)
      dimension ui1(nInv), ui2(nInv*(nInv+1)/2)
      dimension ui3(nInv*(nInv+1)/2), statev(numStatev)
      dimension fieldv(numFieldv), fieldvInc(numFieldv)
      dimension props(numProps)
C
      parameter ( half = 0.5d0,
     *            zero = 0.d0, 
     *            one  = 1.d0, 
     *            two  = 2.d0, 
     *            three= 3.d0, 
     *            four = 4.d0, 
     *            five = 5.d0, 
     *            six  = 6.d0,
c
     *            index_I1 = 1,
     *            index_J  = 3,
     *            asmall   = 2.d-16  )
C
C     3-family fibre model (based on 4-family by Hu, Baek and Humphrey)
C
C     Read material properties
      C10 = props(1)
      rk1 = props(3)
      rk2 = props(4)
      rk3 = props(5)
	  rk4 = props(6)
C
C     Compute Udev and 1st and 2nd derivatives w.r.t invariants
C     Isotropic part of function	
      ua(2) = zero	
C	  
	  ua(2) = ua(2) + C10 * (aInv(1) - three)
	  ui1(1) = C10
	  ui2(indx(1,1)) = zero
C		
C	  Fibre contribution 
C
      do k1 = 1, nFiber
        index_i4 = indxInv4(k1,k1)
		if(kl.gt.one) then
		  E_alpha1 = (aInv(index_i4) - one  )
	      E_alpha = max(E_alpha1, zero)
		  ht4a    = half + sign(half,E_alpha1 + asmall)
		  aux     = exp(rk4*E_alpha*E_alpha)
C
C     deviatoric energy
		  ua(2) = ua(2) +  (rk3 / (four* rk4) * ( aux - one ) )
C			
C	  duDi	
          ui1(index_i4) = ( (rk3 / 2) * aux ) * E_alpha
C
C     d2uDiDi
          ui2(indx(index_i4,index_i4)) = aux * ( rk3 * rk4 * E_alpha*E_alpha + one)
		else
		  E_alpha1 = (aInv(index_i4) - one  )
	      E_alpha = max(E_alpha1, zero)
		  ht4a    = half + sign(half,E_alpha1 + asmall)
		  aux     = exp(rk2*E_alpha*E_alpha)
c
c     deviatoric energy
c
		  ua(2) = ua(2) +  (rk1 / (four* rk2) * ( aux - one ) )
C			
C	  duDi	
          ui1(index_i4) = ( (rk1 / 2) * aux ) * E_alpha
C
C     d2uDiDi
          ui2(indx(index_i4,index_i4)) = aux * ( rk1 * rk2 * E_alpha*E_alpha + one)
		end if
      end do
c     
c     compressible case
      if(props(2).gt.zero) then
         Dinv = one / props(2)
         det = ainv(index_J)
         ua(1) = ua(2) + Dinv *((det*det - one)/two - log(det))
         ui1(index_J) = Dinv * (det - one/det)
         ui2(indx(index_J,index_J))= Dinv * (one + one / det / det)
         if (hybflag.eq.1) then
           ui3(indx(index_J,index_J))= - Dinv * two / (det*det*det)
         end if
      end if
c
      return
      end
C-------------------------------------------------------------
C     Function to map index from Square to Triangular storage 
C 		 of symmetric matrix
C
      integer function indx( i, j )
      include 'aba_param.inc'
      ii = min(i,j)
      jj = max(i,j)
      indx = ii + jj*(jj-1)/2
      return
      end
C-------------------------------------------------------------
C
C     Function to generate enumeration of scalar
C     Pseudo-Invariants of type 4

      integer function indxInv4( i, j )
      include 'aba_param.inc'
      ii = min(i,j)
      jj = max(i,j)
      indxInv4 = 4 + jj*(jj-1) + 2*(ii-1)
      return
      end
C-------------------------------------------------------------
C
C     Function to generate enumeration of scalar
C     Pseudo-Invariants of type 5
C
      integer function indxInv5( i, j )
      include 'aba_param.inc'
      ii = min(i,j)
      jj = max(i,j)
      indxInv5 = 5 + jj*(jj-1) + 2*(ii-1)
      return
      end
C-------------------------------------------------------------