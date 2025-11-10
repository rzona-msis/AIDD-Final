"""
Bookings controller - handles booking creation, approval, and management.

Blueprint: bookings_bp
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, abort
from flask_login import login_required, current_user
from src.forms import BookingForm, ReviewForm
from src.data_access.booking_dal import BookingDAL
from src.data_access.resource_dal import ResourceDAL
from src.data_access.review_dal import ReviewDAL
from datetime import datetime

bookings_bp = Blueprint('bookings', __name__)


@bookings_bp.route('/create', methods=['GET', 'POST'])
@login_required
def create_booking():
    """
    Create a new booking request.
    
    Checks for conflicts and auto-approves if resource doesn't require approval.
    """
    form = BookingForm()
    
    if form.validate_on_submit():
        resource_id = int(form.resource_id.data)
        resource = ResourceDAL.get_resource_by_id(resource_id)
        
        if not resource:
            flash('Resource not found.', 'danger')
            return redirect(url_for('resources.list_resources'))
        
        try:
            # Determine initial status
            status = 'pending' if resource['requires_approval'] else 'approved'
            
            # Create booking
            booking_id = BookingDAL.create_booking(
                resource_id=resource_id,
                requester_id=current_user.user_id,
                start_datetime=form.start_datetime.data.isoformat(),
                end_datetime=form.end_datetime.data.isoformat(),
                status=status,
                notes=form.notes.data
            )
            
            if status == 'approved':
                flash('Booking confirmed! You can view it in your dashboard.', 'success')
            else:
                flash('Booking request submitted! The owner will review your request.', 'info')
            
            return redirect(url_for('bookings.view_booking', booking_id=booking_id))
            
        except ValueError as e:
            flash(str(e), 'danger')
        except Exception as e:
            flash(f'Error creating booking: {str(e)}', 'danger')
    
    # Pre-fill resource_id from query parameter
    resource_id = request.args.get('resource_id')
    if resource_id:
        form.resource_id.data = resource_id
        resource = ResourceDAL.get_resource_by_id(int(resource_id))
        return render_template('bookings/create.html', form=form, resource=resource)
    
    return render_template('bookings/create.html', form=form, resource=None)


@bookings_bp.route('/<int:booking_id>')
@login_required
def view_booking(booking_id):
    """
    View booking details.
    
    Only booking requester, resource owner, or admin can view.
    """
    booking = BookingDAL.get_booking_by_id(booking_id)
    
    if not booking:
        abort(404)
    
    # Check permissions
    if current_user.user_id not in [booking['requester_id'], booking['resource_owner_id']] \
       and not current_user.is_admin():
        abort(403)
    
    # Check if user can leave a review
    can_review = False
    if booking['status'] == 'completed' and current_user.user_id == booking['requester_id']:
        can_review = ReviewDAL.can_user_review(current_user.user_id, booking['resource_id'])
    
    return render_template('bookings/view.html', booking=booking, can_review=can_review)


@bookings_bp.route('/<int:booking_id>/approve', methods=['POST'])
@login_required
def approve_booking(booking_id):
    """
    Approve a pending booking.
    
    Only resource owner or admin can approve.
    """
    booking = BookingDAL.get_booking_by_id(booking_id)
    
    if not booking:
        abort(404)
    
    # Check permissions
    if current_user.user_id != booking['resource_owner_id'] and not current_user.is_admin():
        abort(403)
    
    if booking['status'] != 'pending':
        flash('This booking cannot be approved.', 'warning')
        return redirect(url_for('bookings.view_booking', booking_id=booking_id))
    
    try:
        # Check for conflicts before approving
        if BookingDAL.has_conflict(
            booking['resource_id'],
            booking['start_datetime'],
            booking['end_datetime'],
            exclude_booking_id=booking_id
        ):
            flash('Cannot approve: booking conflicts with another reservation.', 'danger')
        else:
            BookingDAL.update_booking_status(booking_id, 'approved')
            flash('Booking approved successfully!', 'success')
            
            # Create Google Calendar event if user has connected their calendar
            from src.services.google_calendar_service import calendar_service
            from src.data_access.user_dal import UserDAL
            requester = UserDAL.get_user_by_id(booking['requester_id'])
            
            if requester and requester['google_calendar_token']:
                try:
                    credentials = calendar_service.get_credentials(
                        requester['google_calendar_token'],
                        requester['google_calendar_refresh_token'],
                        requester['google_calendar_token_expiry']
                    )
                    
                    if credentials:
                        booking_data = {
                            'booking_id': booking_id,
                            'resource_title': booking['resource_title'],
                            'location': booking['resource_location'],
                            'category': booking['resource_category'],
                            'capacity': booking['resource_capacity'],
                            'start_datetime': booking['start_datetime'],
                            'end_datetime': booking['end_datetime'],
                            'status': 'approved',
                            'notes': booking['notes']
                        }
                        
                        event_id = calendar_service.create_booking_event(credentials, booking_data)
                        
                        if event_id:
                            # Store event ID in booking
                            BookingDAL.update_calendar_event_id(booking_id, event_id)
                            flash('✓ Booking added to your Google Calendar!', 'info')
                except Exception as e:
                    print(f"Failed to create calendar event: {e}")
                    # Don't fail the approval if calendar sync fails
            
            # Log admin action if admin
            if current_user.is_admin():
                from src.data_access.admin_dal import AdminDAL
                AdminDAL.log_action(
                    current_user.user_id,
                    'Approved booking',
                    'bookings',
                    booking_id,
                    f'Booking #{booking_id} for resource #{booking["resource_id"]}'
                )
    
    except Exception as e:
        flash(f'Error approving booking: {str(e)}', 'danger')
    
    return redirect(url_for('bookings.view_booking', booking_id=booking_id))


@bookings_bp.route('/<int:booking_id>/reject', methods=['POST'])
@login_required
def reject_booking(booking_id):
    """
    Reject a pending booking.
    
    Only resource owner or admin can reject.
    """
    booking = BookingDAL.get_booking_by_id(booking_id)
    
    if not booking:
        abort(404)
    
    # Check permissions
    if current_user.user_id != booking['resource_owner_id'] and not current_user.is_admin():
        abort(403)
    
    if booking['status'] != 'pending':
        flash('This booking cannot be rejected.', 'warning')
        return redirect(url_for('bookings.view_booking', booking_id=booking_id))
    
    try:
        BookingDAL.update_booking_status(booking_id, 'rejected')
        flash('Booking rejected.', 'info')
        
        # Log admin action if admin
        if current_user.is_admin():
            from src.data_access.admin_dal import AdminDAL
            AdminDAL.log_action(
                current_user.user_id,
                'Rejected booking',
                'bookings',
                booking_id
            )
    
    except Exception as e:
        flash(f'Error rejecting booking: {str(e)}', 'danger')
    
    return redirect(url_for('bookings.view_booking', booking_id=booking_id))


@bookings_bp.route('/<int:booking_id>/cancel', methods=['POST'])
@login_required
def cancel_booking(booking_id):
    """
    Cancel a booking.
    
    Only booking requester can cancel.
    """
    booking = BookingDAL.get_booking_by_id(booking_id)
    
    if not booking:
        abort(404)
    
    # Check permissions
    if current_user.user_id != booking['requester_id']:
        abort(403)
    
    if booking['status'] in ['completed', 'cancelled', 'rejected']:
        flash('This booking cannot be cancelled.', 'warning')
        return redirect(url_for('bookings.view_booking', booking_id=booking_id))
    
    try:
        BookingDAL.update_booking_status(booking_id, 'cancelled')
        flash('Booking cancelled successfully.', 'success')
    except Exception as e:
        flash(f'Error cancelling booking: {str(e)}', 'danger')
    
    return redirect(url_for('dashboard.my_bookings'))


@bookings_bp.route('/<int:booking_id>/review', methods=['GET', 'POST'])
@login_required
def create_review(booking_id):
    """
    Create a review for a completed booking.
    
    Only booking requester can review after completion.
    """
    booking = BookingDAL.get_booking_by_id(booking_id)
    
    if not booking:
        abort(404)
    
    # Check permissions
    if current_user.user_id != booking['requester_id']:
        abort(403)
    
    if booking['status'] != 'completed':
        flash('You can only review completed bookings.', 'warning')
        return redirect(url_for('bookings.view_booking', booking_id=booking_id))
    
    # Check if already reviewed
    if not ReviewDAL.can_user_review(current_user.user_id, booking['resource_id']):
        flash('You have already reviewed this resource.', 'info')
        return redirect(url_for('resources.view_resource', resource_id=booking['resource_id']))
    
    form = ReviewForm()
    
    if form.validate_on_submit():
        try:
            ReviewDAL.create_review(
                resource_id=booking['resource_id'],
                reviewer_id=current_user.user_id,
                rating=int(form.rating.data),
                comment=form.comment.data,
                booking_id=booking_id
            )
            
            flash('Review submitted successfully! Thank you for your feedback.', 'success')
            return redirect(url_for('resources.view_resource', resource_id=booking['resource_id']))
            
        except Exception as e:
            flash(f'Error submitting review: {str(e)}', 'danger')
    
    # Pre-populate form
    if request.method == 'GET':
        form.resource_id.data = booking['resource_id']
        form.booking_id.data = booking_id
    
    return render_template('bookings/review.html', form=form, booking=booking)


@bookings_bp.route('/check-availability')
@login_required
def check_availability():
    """
    API endpoint to check booking availability.
    
    Returns JSON with conflict status.
    """
    resource_id = request.args.get('resource_id', type=int)
    start_datetime = request.args.get('start_datetime')
    end_datetime = request.args.get('end_datetime')
    
    if not all([resource_id, start_datetime, end_datetime]):
        return jsonify({'error': 'Missing parameters'}), 400
    
    try:
        has_conflict = BookingDAL.has_conflict(resource_id, start_datetime, end_datetime)
        return jsonify({
            'available': not has_conflict,
            'message': 'Time slot is available' if not has_conflict else 'Time slot already booked'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

